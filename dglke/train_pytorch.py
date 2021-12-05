# -*- coding: utf-8 -*-
#
# train_pytorch.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.multiprocessing as mp
from multiprocessing import Lock
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th
import numpy as np
import itertools 
import copy
import sys

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from .models.pytorch.tensor_models import thread_wrapped_func
from .models import KEModel
from .utils import save_model, get_compatible_batch_size

import os
import logging
import time
from functools import wraps
import random
from collections import Counter

import dgl
from dgl.contrib import KVClient
import dgl.backend as F
from dgl.network import _send_kv_msg, _recv_kv_msg, _clear_kv_msg, KVMsgType, KVStoreMsg
from dgl.data.utils import load_graphs
from .dataloader import ChunkNegEdgeSubgraph
from dgl.contrib.sampling.sampler import EdgeSubgraph, EdgeSampler

from .dataloader import EvalDataset, create_neg_subgraph
from .dataloader import get_dataset

class Forwardsubgraph:
    def __init__(self, ndata, edata, head_ids, tail_ids, num_chunks=None, chunk_size=None, neg_sample_size=None, neg_head=None):
        self.ndata = {}
        self.edata = {}
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head
        self.head_ids = head_ids
        self.tail_ids = tail_ids

        self.ndata['id'] = ndata
        if edata != None:
            self.edata['id'] = edata

    def all_edges(order='eid'):
        return self.head_ids, tail_ids


class KGEClient(KVClient):
    """User-defined kvclient for DGL-KGE
    """
    # new code
    def __init__(self, server_namebook):
        super(KGEClient, self).__init__(server_namebook)
        self.entity_cache = set()
        self.relation_cache = set()
        self.entity_num = 0
        self.relation_num = 0


    def _pull_handler(self, name, ID, target):
        """Default handler for PULL operation.

        On default, _pull_handler perform get operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list.
        target : dict of data
            self._data_store

        Return
        ------
        tensor
            a tensor with the same row size of ID.
        """

        original_name = name[0:-6]
        local_num = target[original_name+'-num-'][0].item()
        cache_num = target[original_name+'-cache_num-'][0].item()

        local_index = th.where(ID < local_num)[0]
        local_ID = ID[local_index]
        remote_index = th.where(ID >= local_num)[0]
        remote_ID = ID[remote_index] - local_num

        # 拉取 local data
        local_data_tensor = target[original_name+'-data-'][local_ID]
        loca_state_sum = target[original_name+'_state-data-'][local_ID]
        # shape0 = data_tensor.shape[0]
        local_result = th.cat((local_data_tensor, loca_state_sum.reshape(local_data_tensor.shape[0], 1)), 1)

        # 拉取cache data 
        buffer_id = self.get_buffer_id()
        cache_data_tensor = target[original_name+'-buffer-'][buffer_id][remote_ID]
        cache_state_sum = target[original_name+'_state-buffer-'][buffer_id][remote_ID]
        cache_result = th.cat((cache_data_tensor, cache_state_sum.reshape(cache_data_tensor.shape[0], 1)), 1)

        # 构造最终结果
        result = th.zeros(ID.shape[0], local_result.shape[1])
        result[local_index] = local_result
        result[remote_index] = cache_result

        return result


    def _push_handler(self, name, ID, data, target, local_cache):
        """Row-Sparse Adagrad updater
            ID is position id
        """
        
        original_name = name[0:-6]
        local_num = target[original_name+'-num-'][0].item()
        cache_num = target[original_name+'-cache_num-'][0].item()

        local_index = th.where(ID < local_num)[0]
        local_ID = ID[local_index]
        local_data = data[local_index]
        remote_index = th.where(ID >= local_num)[0]
        remote_ID = ID[remote_index] - local_num
        remote_data = data[remote_index]
        
        # 更新local data
        local_state_sum = target[original_name+'_state-data-']
        local_grad_sum = (local_data * local_data).mean(1) # 梯度平方的平均值
        local_state_sum.index_add_(0, local_ID, local_grad_sum)
        local_std = local_state_sum[local_ID]  # _sparse_mask
        local_std_values = local_std.sqrt_().add_(1e-10).unsqueeze(1)
        local_tmp = (-self.clr * local_data / local_std_values)
        target[name].index_add_(0, local_ID, local_tmp)

        # 更新cache data
        buffer_id = self.get_buffer_id()
        remote_state_sum = target[original_name+'_state-buffer-'][buffer_id]
        remote_grad_sum = (remote_data * remote_data).mean(1) # 梯度平方的平均值
        remote_state_sum.index_add_(0, remote_ID, remote_grad_sum)
        remote_std = remote_state_sum[remote_ID]  # _sparse_mask
        remote_std_values = remote_std.sqrt_().add_(1e-10).unsqueeze(1)
        remote_tmp = (-self.clr * remote_data / remote_std_values)
        target[original_name+'-buffer-'][buffer_id].index_add_(0, remote_ID, remote_tmp)

        if local_cache == True:
            # emb_num = target[original_name+'-num-'][0].item()
            # cache_num = target[original_name+'-cache_num-'][0].item()
            # cache_index = th.where(ID >= emb_num)[0]
            # ID = ID[cache_index] - emb_num
            one_tensor = th.ones(remote_ID.shape[0], dtype=th.long)
            target[original_name+'-cache_hit-'].index_add_(0, remote_ID, one_tensor)

            # if self._client_id == 0:
            #     hit_tensor = self._data_store[original_name+'-cache_hit-'][:cache_num]
            #     hit_index = th.where(hit_tensor > 0)[0]
            #     hit_num = hit_index.shape[0]

                # print("[hit ratio]", self._client_id , name, "hit num:", hit_num, "cache num:", cache_num, "percent:", hit_num/cache_num)

            # LFU 缓存置换
            # cache_hit_tensor = target[original_name+'-cache_hit-'][:cache_num]
            # global_ID = target[original_name+'-cache-'][:cache_num]
            # DATA = target[original_name+'-data-'][emb_num: emb_num+cache_num]
            # STATE = target[original_name+'_state-data-'][emb_num: emb_num+cache_num]

            # sorted_id = F.tensor(np.argsort(F.asnumpy(-cache_hit_tensor)))
            # cache_hit_tensor = cache_hit_tensor[sorted_id]
            # global_ID = global_ID[sorted_id]
            # DATA = DATA[sorted_id]
            # STATE = STATE[sorted_id]

            # target[original_name+'-cache_hit-'][:cache_num] = cache_hit_tensor
            # target[original_name+'-cache-'][:cache_num] = global_ID
            # target[original_name+'-data-'][emb_num: emb_num+cache_num] = DATA
            # target[original_name+'_state-data-'][emb_num: emb_num+cache_num] = STATE

            # local_ID = global_ID.clone()
            # if (original_name+'-g2l-' in self._has_data) == True:
            #     local_ID = target[original_name+'-g2l-'][local_ID]
            # cache_list = list(range(cache_num))
            # cache_tensor = th.tensor(cache_list, dtype=th.long) + emb_num
            # target[original_name+'-l2p-'][local_ID] = cache_tensor
 


    def _cache_handler(self, name, ID, state, data, target, g2l, add):
        """Cache remote embeddings
            缓存替换策略先进先出（LFU），50000是缓存列表大小
            存储方案：-data- 本地参数+缓存参数（50000）
                    -grad- 缓存参数变化量和（50000）
                    ID: global ID
        """
      
        emb_num = target[name+'-num-'][0].item()
        cache_num = target[name+'-cache_num-'][0].item()
        add_num = ID.shape[0]

        target[name+'-cache-part-'][ID] = self._machine_id

        add_list = list(range(0, add_num))
        add_tensor = th.tensor(add_list, dtype=th.long)
        # add_tensor = add_tensor + cache_num
        print("cache handler", add_tensor.shape, ID.shape)
        target[name+'-cache-'][add_tensor] = ID

        local_ID = ID
        if g2l == True:
            local_ID = target[name+'-g2l-'][ID]
      
        target[name+'-l2p-'][local_ID] = add_tensor + emb_num
        cache_num += add_num
        target[name+'-cache_num-'][0] = cache_num
        # else:
        #     # TODO
        #     swap_num = ID.shape[0]
        #     if swap_num > int(cache_num * 0.9):
        #         swap_num = int(cache_num * 0.9)
        #         ID = ID[-swap_num:]
        #         local_ID = local_ID[-swap_num:]
        #         data = data[-swap_num:]
        #         state = state[-swap_num:]
        #     swap_list = list(range(swap_num))
        #     swap_tensor = th.tensor(swap_list, dtype=th.long)
        #     swap_tensor = swap_tensor + cache_num - swap_num
        #     swap_global_ID = target[name+'-cache-'][swap_tensor]
        #     target[name+'-cache-part-'][swap_global_ID] = target[name+'-part-'][swap_global_ID]
        #     target[name+'-cache-'][swap_tensor] = ID
        #     target[name+'-l2p-'][local_ID] = swap_tensor + emb_num
        #     # target[name+'-cache_num-'][0] = cache_num

        position_ID = target[name+'-l2p-'][local_ID]
        buffer_id = self.get_buffer_id()
        
        target[name+'-buffer-'][buffer_id][position_ID - emb_num] = data
        target[name+'_state-buffer-'][buffer_id][position_ID - emb_num] = state   # state_sum实际值
        # 清空缓存命中值
        target[name+'-cache_hit-'][position_ID - emb_num] = 0

        # for i in range(add_num):
        #     if target[name+'-cache-part-'][ID[i]] != self._machine_id:
        #         remote_after += 1 
        #     else:
        #         local_after += 1

        # print("[after cache handler]", name, cache_num, target[name+'-cache_num-'][0], local_after, remote_after)
        

    def _update_cache(self, name, ID, state, data, target):
        """Update cached entity embeddings
            ID: position id
        """

        # cache_index = 0
        # emb_num = target[name+'-num-']
        # if name == 'entity_emb':
        #     target[name+'_state-data-'][ID+10000] = 0
        # else:
        #     target[name+'_state-data-'][ID+emb_num] = 0
    
        # 将缓存的梯度和版本号清空
        # target[name+'-grad-'][:] = 0
        # 更新缓存参数和梯度平方和
        emb_num = target[name+'-num-'][0].item()
        buffer_id = self.get_buffer_id()
        buffer_id = buffer_id ^ 1

        target[name+'-buffer-'][buffer_id][ID - emb_num] = data
        target[name+'_state-buffer-'][buffer_id][ID - emb_num] = state
        target[name+'-cache_hit-'][:]= 0


    def _clear_cache(self):
        """Clear cached entity embeddings
        """

        entity_index = self._data_store['entity_emb-num-'][0]
        self._data_store['entity_emb-cache_num-'][:] = 0
        self._data_store['entity_emb-cache-part-'][:] = self._data_store['entity_emb-part-'][:]
        self._data_store['entity_emb_state-data-'][entity_index: ] = 0
        # 如果不清理关系缓存，则注释掉下面这一句
        self._data_store['relation_emb-cache-part-'][:] = self._data_store['relation_emb-part-'][:]


    def cache_sync(self, name):
        """Synchronize data between KVServer and local cache.

        Note that cache_sync() is an async operation that will return immediately after calling.

        Parameters
        ----------
        name : str
            data name

        """
        assert len(name) > 0, 'name cannot be empty.'

        for msg in self._garbage_msg:
            _clear_kv_msg(msg)
        self._garbage_msg = []

        id_tensor = None
        data_tensor = None
        partition_start = time.time()
        
        # new code push data to kvserver and get the lastest parameters
        emb_num = self._data_store[name+'-num-'][0].item()
        cache_num = self._data_store[name+'-cache_num-'][0].item()
        # print("[cache sync]", name, self._data_store[name+'-cache_num-'][0])
        id_tensor = self._data_store[name+'-cache-'][:cache_num]

        # 10000 is the max cache len
        # start = 0
        # end = cache_num
        # # if id_tensor[cache_num] < 0:
        # #     end = cache_num
        # id_tensor = id_tensor[start:end]
        # id_tensor = th.tensor(range(emb_num))

        # partition data 
        machine_id = self._data_store[name+'-part-'][id_tensor]

        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        id_tensor = id_tensor[sorted_id]
        # data_tensor = data_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        # push data to server by order
        partition_time = time.time() - partition_start

        # sync data from server by order
        start = 0
        pull_count = 0
        local_id = None
        cache_pull_id = None 
        hit_num = 0
        # cache_id_tensor = th.tensor([], dtype=th.long)
        # cache_list = []
        remote_num = 0

        send_start=time.time()
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            # partial_data = data_tensor[start:end]
            if machine[idx] == self._machine_id: # no local pull
                continue
            else: # push data to remote server 
                remote_num += partial_id.shape[0]
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id] # global id -> local id的映射
                else:
                    local_id = partial_id
                # lock.acquire()
                # print("[start I get the lock sync]")
                if (name+'-l2p-' in self._has_data) == True:
                    local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射
                # time.sleep(1)
                # print("end I return the lock sync")
                # lock.release()
                # 记录缓存命中率
                hit_tensor = self._data_store[name+'-cache_hit-'][local_id - emb_num] 
                hit_index = th.where(hit_tensor > 0)[0]
                # print("[hit]", machine[idx], hit_index.shape[0], id_tensor.shape[0], hit_num)
                hit_num += hit_index.shape[0]

                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    # state=partial_state_sum,
                    data=None,
                    c_ptr=None)

                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)
                pull_count += 1

            start += count[idx]
        send_time = time.time()-send_start 
        ratio = 0.0
        if cache_num > 0:
            ratio = hit_num/cache_num

        # if self._client_id == 0:
        # print("[hit ratio]", self._client_id , name, "hit num:", hit_num, "cache num:", cache_num, "remote: send", remote_num, "percent:", ratio)
        
        msg_list = []
        # cache_list = []

        remote_start = time.time()
        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            msg_list.append(remote_msg)
            # cache_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)
            # print("[recv msg]", idx)
        remote_time = time.time()-remote_start

        merge_start = time.time()
        # sort msg by server id and merge tensor together
        if len(msg_list) > 0:
            # print("[sync msg list]", len(msg_list), pull_count)
            msg_list.sort(key=self._takeId)
            cache_id_tensor = F.cat(seq=[msg.id for msg in msg_list], dim=0)
            cache_data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)
            data_tensor = cache_data_tensor[:, :-1]
            state_sum_tensor = cache_data_tensor[:, -1]
            # 更新本地缓存
            if (name+'-g2l-' in self._has_data) == True:
                local_id = self._data_store[name+'-g2l-'][cache_id_tensor] # global id -> local id的映射
            else:
                local_id = cache_id_tensor
            
            # lock.acquire()
            if (name+'-l2p-' in self._has_data) == True:
                local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射
            # print("[after sync]", local_id.shape, cache_data_tensor.shape)
            
            self._update_cache(name, local_id, state_sum_tensor, data_tensor, self._data_store)
            # lock.release()
        local_time = time.time() - merge_start
        
        # print("partition_time: ", partition_time, "send_time: ", send_time, "remote_time: ", remote_time, "update time:", local_time)

        return partition_time+local_time, send_time+remote_time, ratio


    def cache_push(self, name, id_tensor, data_tensor, local_cache):
        """Push data to KVServer and local cache.

        Note that cache_push() is an async operation that will return immediately after calling.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor (mx.ndarray or torch.tensor)
            a vector storing the global data ID
        data_tensor : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of data ID
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        assert F.shape(id_tensor)[0] == F.shape(data_tensor)[0], 'The data must has the same row size with ID.'

        # print("[cache push start global]", name, id_tensor.shape, id_tensor, data_tensor.shape, local_cache, synchronize)

        local_id = None
        partition_start = time.time()
        # new code push data to local kvserver or local cache
        machine_id = self._data_store[name+'-part-'][id_tensor]

        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
        id_tensor = id_tensor[sorted_id]
        data_tensor = data_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        # push data to server by order
        # print("start sleep")
        # time.sleep(5)
        # print("finish sleep")

        # print(self._machine_id, "[push send]", count[0], count[1], count[2], count[3])
        
        partition_time = time.time() - partition_start
        send_start = time.time()

        start = 0
        local_data = None
        remote_time = 0
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            partial_data = data_tensor[start:end]
            if machine[idx] == self._machine_id: # local push
                # Note that DO NOT push local data right now because we can overlap
                # communication-local_push here
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id]
                else:
                    local_id = partial_id
                local_data = partial_data
            else: # push data to remote server

                msg = KVStoreMsg(
                    type=KVMsgType.PUSH, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    data=partial_data,
                    c_ptr=None)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)

            start += count[idx]
        send_time = time.time() - send_start

        local_start = time.time()

        # 在push的时候同时更新cache
        # if local_cache == True: 
        #     id_tensor = id_tensor[back_sorted_id]
        #     data_tensor = data_tensor[back_sorted_id]
        #     machine_id = self._data_store[name+'-cache-part-'][id_tensor]
        #     sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        #     back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
        #     id_tensor = id_tensor[sorted_id]
        #     data_tensor = data_tensor[sorted_id]
        #     machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)

        #     start = 0
        #     partial_id = None
        #     partial_data = None
        #     for idx in range(len(machine)):
        #         end = start + count[idx]
        #         if start == end: # No data for target machine
        #             continue
        #         partial_id = id_tensor[start:end]
        #         partial_data = data_tensor[start:end]
        #         if machine[idx] == self._machine_id: 
        #             if (name+'-g2l-' in self._has_data) == True:
        #                 partial_id = self._data_store[name+'-g2l-'][partial_id]
        #             # lock.acquire()
        #             if (name+'-l2p-' in self._has_data) == True: 
        #                 partial_id = self._data_store[name+'-l2p-'][partial_id] # local id -> position id的映射
        #             break

        #     self._push_handler(name+'-data-', partial_id, partial_data, self._data_store, local_cache) # 调用push
        #     # lock.release()
        #             # break
            
        # # 不更新缓存，只更新本地参数
        if local_id is not None:
            sorted_local_id = F.tensor(np.argsort(F.asnumpy(local_id)))
            local_id = local_id[sorted_local_id]
            local_data = local_data[sorted_local_id]
             # local push
                # if (name+'-g2l-' in self._has_data) == True:
                #     local_id = self._data_store[name+'-g2l-'][local_id]
            if (name+'-l2p-' in self._has_data) == True:
                local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射
            self._push_handler(name+'-data-', local_id, local_data, self._data_store, local_cache)

        local_time = time.time() - local_start
        return partition_time+local_time, send_time


    def cache_pull(self, name, id_tensor, local_cache):
        """Pull message from KVServer and local cache.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor (mx.ndarray or torch.tensor)  global id
            a vector storing the ID list

        Returns
        -------
        tensor
            a data tensor with the same row size of id_tensor.
        """

        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'

        local_num = 0
        cache_hit_ratio = 0.0
        start1 = time.time()
        for msg in self._garbage_msg:
            _clear_kv_msg(msg)
        self._garbage_msg = []

        # print("[pull] id tensor", id_tensor.shape)

        cache_num = self._data_store[name+'-cache_num-'][0].item()
        # machine_id = self._data_store[name+'-cache-part-'][id_tensor]
        machine_id = self._data_store[name+'-part-'][id_tensor]
        
        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
        id_tensor = id_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        partition_time = time.time()-start1

        start = 0
        pull_count = 0
        local_id = None
        cache_pull_id = None 
        # cache_id_tensor = th.tensor([], dtype=th.long)
        cache_list = []
        local_num = 0
        remote_num = 0

        send_start=time.time()
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            if machine[idx] == self._machine_id: # local pull
                # Note that DO NOT pull local data right now because we can overlap
                # local_pull here
                if cache_num>0:
                    partial_cache_id = self._data_store[name+'-part-'][partial_id]
                    cache_index = th.where(partial_cache_id != self._machine_id)[0]
                    cache_hit_ratio = cache_index.shape[0]/id_tensor.shape[0]

                local_num += partial_id.shape[0]
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id]
                else:
                    local_id = partial_id
            else: # pull data from remote server
                # cache_id_tensor = th.cat((cache_id_tensor, partial_id), 0)
                remote_num += partial_id.shape[0]
                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=partial_id,
                    data=None,
                    c_ptr=None)
                # print("[client global id]", msg.rank, msg.id)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)
                pull_count += 1

            start += count[idx]
        send_time = time.time()-send_start 
        # print("machine: ", self._machine_id, "client: ", self._client_id,"[sum]", id_tensor.shape[0], "[local]", local_num, "remote", remote_num)
        
        local_start = time.time() 
        msg_list = []
        if local_id is not None: # local pull
            # lock.acquire()
            if (name+'-l2p-' in self._has_data) == True:
                local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射
            local_data = self._pull_handler(name+'-data-', local_id, self._data_store)
            # lock.release()
            s_id = random.randint(self._machine_id*self._group_count, (self._machine_id+1)*self._group_count-1)
            local_msg = KVStoreMsg(
                type=KVMsgType.PULL_BACK, 
                rank=s_id,
                name=name, 
                id=None,
                data=local_data,
                c_ptr=None)
            msg_list.append(local_msg)
            self._garbage_msg.append(local_msg)

        local_time = time.time()-local_start

        remote_start = time.time()
        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            msg_list.append(remote_msg)
            cache_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)
        remote_time = time.time()-remote_start
        # remote_time = 0

        merge_start = time.time()
        # if pull_count > 0:
            # sort msg by server id and merge tensor together
        msg_list.sort(key=self._takeId)
        data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)
        data_tensor = data_tensor[:, :-1]
        local_time += time.time()-merge_start

        # print("[pull] data tensor", data_tensor.shape)
        
        # if (pull_count>0) and (local_cache == True):
        #     cache_list.sort(key=self._takeId)
        #     cache_tensor = F.cat(seq=[msg.data for msg in cache_list], dim=0)
        #     cache_id_tensor = F.cat(seq=[msg.id for msg in cache_list], dim=0)
        #     grad_tensor = cache_tensor[:, :-1]
        #     state_sum_tensor = cache_tensor[:, -1]
            
        #     g2l = False
        #     if (name+'-g2l-' in self._has_data) == True:
        #         g2l = True
        #     # lock.acquire()
        #     self._cache_handler(name, cache_id_tensor, state_sum_tensor, grad_tensor, self._data_store, g2l, add=False)
        #     # lock.release()
        return data_tensor[back_sorted_id], partition_time+local_time, remote_time+send_time, remote_num/id_tensor.shape[0], cache_hit_ratio, local_num, remote_num # return data with original index order
        
        # target[name][ID] = data

    def cache_topk(self, entity_tensor, relation_tensor, entity_numlist, relation_numlist, topk=0, ent_topk=-1, rel_topk=-1):
        # assert len(name) > 0, 'name cannot be empty.'
        # assert F.ndim(id_tensor) == 1, 'ID must be a vector.'

        # sum_num = id_tensor.shape[0]
        

        entity_machine_id = self._data_store['entity_emb-part-'][entity_tensor]
        relation_machine_id = self._data_store['relation_emb-part-'][relation_tensor]

        cache_id_list = []
        machine_id_list = []
        remote_num = 0
        local_num = 0

        # local_index = th.where(machine_id == self._machine_id)[0]
        # local_num = local_index.shape[0]
        entity_remote_index = th.where(entity_machine_id != self._machine_id)[0]
        entity_remote_num = entity_remote_index.shape[0]
        entity_tensor = entity_tensor[entity_remote_index]
        entity_idlist = entity_tensor.tolist()
        entity_remote_numlist = [entity_numlist[i] for i in entity_remote_index.tolist()]

        relation_remote_index = th.where(relation_machine_id != self._machine_id)[0]
        relation_remote_num = relation_remote_index.shape[0]
        relation_tensor = relation_tensor[relation_remote_index]
        relation_idlist = relation_tensor.tolist()
        relation_remote_numlist = [relation_numlist[i] for i in relation_remote_index.tolist()]

        # print("remote entity id list", len(entity_idlist), len(relation_idlist))
        # print("remote entity list", len(entity_remote_numlist), len(relation_remote_numlist))

        # cache_num = min(, sum_num)
        cache_num = min(topk, 50000)

        entity_result = []
        relation_result = []
        # id_tensor = id_tensor[remote_index]
        i = 0
        j = 0
        k = 0
        ent_range = len(entity_idlist)
        rel_range = len(relation_idlist)
        if ent_topk >= 0:
            ent_range = min(len(entity_idlist), ent_topk)
        if rel_topk >= 0:
            rel_range = min(len(relation_idlist), rel_topk)
        
        while i < ent_range and j < rel_range and k < topk:
            if entity_remote_numlist[i] > relation_remote_numlist[j]:
                entity_result.append(entity_idlist[i])
                i += 1
            else:
                relation_result.append(relation_idlist[j])
                j += 1
            k += 1
        while i < ent_range and k < topk:
            entity_result.append(entity_idlist[i])
            i += 1
            k += 1
        while j < rel_range and k < topk:
            relation_result.append(relation_idlist[j])
            j += 1
            k += 1

        entity_result.sort()
        relation_result.sort()
        # id_tensor = id_tensor[:cache_num]
        # id_tensor = F.tensor(np.unique(F.asnumpy(id_tensor)))
        # id_tensor = id_tensor[sorted_id]
        # print("[cache init] entity count: ", len(entity_result))
        # print("[cache init] relation count: ", len(relation_result))

        entity_result_tensor = th.tensor(entity_result, dtype=th.long)
        # entity_result_tensor = F.tensor(np.unique(F.asnumpy(entity_result_tensor)))
        relation_result_tensor = th.tensor(relation_result, dtype=th.long)
        # relation_result_tensor = F.tensor(np.unique(F.asnumpy(relation_result_tensor)))

        return entity_result_tensor, relation_result_tensor

    def cache_init(self, name, id_tensor, topk):
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'

        for msg in self._garbage_msg:
            _clear_kv_msg(msg)
        self._garbage_msg = []

        sum_num = id_tensor.shape[0]
        
        machine_id = self._data_store[name+'-part-'][id_tensor]

        cache_id_list = []
        machine_id_list = []
        remote_num = 0
        local_num = 0

        local_index = th.where(machine_id == self._machine_id)[0]
        local_num = local_index.shape[0]
        remote_index = th.where(machine_id != self._machine_id)[0]
        remote_num = remote_index.shape[0]

        id_tensor = id_tensor[remote_index]

        # for i in range(sum_num):
        #     if machine_id[i] != self._machine_id:
        #         cache_id_list.append(id_tensor[i])
        #         machine_id_list.append(machine_id[i])
        #         remote_num += 1

        # cache_num = int(remote_num*ratio)
        # cache_num = remote_num - 40
        cache_num = min(topk, sum_num)
        cache_num = min(cache_num, 50000)

        id_tensor = id_tensor[:cache_num]
        machine_id = self._data_store[name+'-part-'][id_tensor]

        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
        id_tensor = id_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        print("[cache init]", machine , " count: ", count)

        # pull data from server by order
        start = 0
        pull_count = 0
        local_id = None
        cache_pull_id = None 
        # cache_id_tensor = th.tensor([], dtype=th.long)
        cache_list = []
        msg_list = []

        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            if machine[idx] == self._machine_id: # local pull
                # local_num += partial_id.shape[0]
                print("error there is no local id")
                continue
            else: # pull data from remote server
                # cache_num2 += partial_id.shape[0]
                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=partial_id,
                    data=None,
                    c_ptr=None)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)
                pull_count += 1
            start += count[idx]

        print("[cache init]", self._machine_id, "client: ", self._client_id, name, "local:", local_num, "remote: ", remote_num, "cached: ", id_tensor.shape[0])

        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            # msg_list.append(remote_msg)
            cache_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)

        # msg_list.sort(key=self._takeId)
        # data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)
        # data_tensor = data_tensor[:, :-1]

        # sort msg by server id and merge tensor together
        if pull_count>0:
            cache_list.sort(key=self._takeId)
            cache_tensor = F.cat(seq=[msg.data for msg in cache_list], dim=0)
            cache_id_tensor = F.cat(seq=[msg.id for msg in cache_list], dim=0)
            grad_tensor = cache_tensor[:, :-1]
            state_sum_tensor = cache_tensor[:, -1]
            # print("after init", cache_id_tensor.shape[0], grad_tensor.shape[0], state_sum_tensor.shape[0], cache_tensor.shape[0])
            
            g2l = False
            if (name+'-g2l-' in self._has_data) == True:
                g2l = True
            self._cache_handler(name, cache_id_tensor, state_sum_tensor, grad_tensor, self._data_store, g2l, add=True)

        # result = 0
        # local_res = 0
        # for i in range(self._data_store[name+'-cache-part-'].shape[0]):
        #     if self._data_store[name+'-cache-part-'][i] != self._machine_id:
        #         result += 1
        #     else:
        #         local_res += 1
        # print("[after cache still remote]", result, local_res)

        return cache_num


    def get_switch_code(self, name):
        return self._data_store[name+"-async_switch-"][0]

    
    def set_switch_code(self, name, switch_code, add=False):
        if add == True:
            self._data_store[name+"-async_switch-"][0] += 1
        else:
            self._data_store[name+"-async_switch-"][0] = switch_code

    def set_buffer_id(self, id):
        self._data_store["relation_emb-async_switch-"][0] = id

    def get_buffer_id(self):
        return self._data_store["relation_emb-async_switch-"][0]

    def set_async_step(self, id=1, add=True):
        if add == True:
            self._data_store["relation_emb-async_switch-"][1] += 1
        else:
            self._data_store["relation_emb-async_switch-"][1] = id

    def get_async_step(self):
        return self._data_store["relation_emb-async_switch-"][1]
    

    # 建立线段树，用于查询
    def build_tree(self, name, pos, l, r):
        """ 0 区间最值,
            1 区间最值位置
            2 左区间下标
            3 右区间下标
            4 懒标记

        """
        self._data_store[name+"-cache_hit-"][pos][0] = 0
        self._data_store[name+"-cache_hit-"][pos][1] = r
        self._data_store[name+"-cache_hit-"][pos][2] = l
        self._data_store[name+"-cache_hit-"][pos][3] = r
        self._data_store[name+"-cache_hit-"][pos][4] = 0     
        if l == r:
            return
        mid = (l+r)//2
        self.build_tree(name, 2*pos+1, l, mid)
        self.build_tree(name, 2*pos+2, mid+1, r)

    
    def push_down_lazy(self, name, pos, target):
        # lazy传递到左右子树
        target[name+"-cache_hit-"][pos*2+1][4] += target[name+"-cache_hit-"][pos][4]
        target[name+"-cache_hit-"][pos*2+2][4] += target[name+"-cache_hit-"][pos][4]
        # lazy值传递至左右子树的最值
        target[name+"-cache_hit-"][pos*2+1][0] += target[name+"-cache_hit-"][pos][4] * (target[name+"-cache_hit-"][pos*2+1][3] - target[name+"-cache_hit-"][pos*2+1][2] + 1)
        target[name+"-cache_hit-"][pos*2+2][0] += target[name+"-cache_hit-"][pos][4] * (target[name+"-cache_hit-"][pos*2+2][3] - target[name+"-cache_hit-"][pos*2+2][2] + 1)
        # 清空lazy
        target[name+"-cache_hit-"][pos][4] = 0


    def cache_search(self, name, pos, ql, qr, target):
        nowl = target[name+"-cache_hit-"][pos][2].item()
        nowr = target[name+"-cache_hit-"][pos][3].item()
        nowmid = (nowl + nowr)//2
        if ql <= nowl and nowr <= qr:
            return target[name+"-cache_hit-"][pos][0], target[name+"-cache_hit-"][pos][1]
        min_value = 1e10
        min_pos = nowr
        if target[name+"-cache_hit-"][pos][4] > 0:
            self.push_down_lazy(name, pos, target)
        if ql <= nowmid:
            min_l, pos_l = self.cache_search(name, pos*2+1, ql, nowmid, target)
            if min_l < min_value:
                min_value = min_l
                min_pos = pos_l
        if qr > nowmid + 1:
            min_r, pos_r = self.cache_search(name, pos*2+2, nowmid+1, qr, target)
            if min_r <= min_value:
                min_value = min_r
                min_pos = pos_r
        return min_value, min_pos


    def cache_add_hit(self, name, pos, al, ar, val, target):
        nowl = target[name+"-cache_hit-"][pos][2].item()
        nowr = target[name+"-cache_hit-"][pos][3].item()
        nowmid = (nowl + nowr)//2

        if al <= nowl and nowr <= ar:
            target[name+"-cache_hit-"][pos][4] += val
            target[name+"-cache_hit-"][pos][0] += val * (target[name+"-cache_hit-"][pos][3] - target[name+"-cache_hit-"][pos][2] + 1)
            return
        if target[name+"-cache_hit-"][pos][4] > 0:
            self.push_down_lazy(name, pos, target)

        if ar <= nowmid:
            self.cache_add_hit(name, pos*2+1, al, ar, val, target)
        elif al > nowmid:
            self.cache_add_hit(name, pos*2+2, al, ar, val, target)
        else:
            self.cache_add_hit(name, pos*2+1, al, nowmid, val, target)
            self.cache_add_hit(name, pos*2+2, nowmid+1, ar, val, target)
        # 维护区间最小值
        target[name+"-cache_hit-"][pos][0] = target[name+"-cache_hit-"][pos*2+1][0]
        target[name+"-cache_hit-"][pos][1] = target[name+"-cache_hit-"][pos*2+1][1]
        if target[name+"-cache_hit-"][pos*2+2][0] <= target[name+"-cache_hit-"][pos][0]:
            target[name+"-cache_hit-"][pos][0] = target[name+"-cache_hit-"][pos*2+2][0]
            target[name+"-cache_hit-"][pos][1] = target[name+"-cache_hit-"][pos*2+2][1]


    def async_pull(self, name, lock):
        
        assert len(name) > 0, 'name cannot be empty.'
        
        for msg in self._garbage_msg:
            _clear_kv_msg(msg)
        self._garbage_msg = []

        id_tensor = None
        data_tensor = None
        partition_start = time.time()
        
        # new code push data to kvserver and get the lastest parameters
        emb_num = self._data_store[name+'-num-'].item()
        cache_num = self._data_store[name+'-cache_num-'][0].item()
        # if name == 'entity_emb':
        id_tensor = self._data_store[name+'-cache-'][:cache_num]

        # id_tensor = id_tensor[:cache_num]

        # partition data 
        machine_id = self._data_store[name+'-part-'][id_tensor]
        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        id_tensor = id_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        
        partition_time = time.time() - partition_start

        # sync data from server by order
        start = 0
        pull_count = 0
        local_id = None
        cache_pull_id = None 
        hit_num = 0
        # local_num = 0
        # remote_num = 0

        send_start=time.time()
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            # partial_data = data_tensor[start:end]
            if machine[idx] == self._machine_id: # no local pull
                continue
            else: # push data to remote server 
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id] # global id -> local id的映射
                else:
                    local_id = partial_id
                # lock.acquire()
                # print("[start I get the lock sync]")
                if (name+'-l2p-' in self._has_data) == True:
                    local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射
                # lock.release()
                # 记录缓存命中率
                hit_tensor = self._data_store[name+'-cache_hit-'][local_id - emb_num] 
                hit_index = th.where(hit_tensor > 0)[0]
                hit_num += hit_index.shape[0]

                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    # state=partial_state_sum,
                    data=None,
                    c_ptr=None)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)
                pull_count += 1

            start += count[idx]
        ratio = 0.0
        if cache_num > 0:
            ratio = hit_num/cache_num
        # print("[hit ratio]", name, "hit num:", hit_num, "cache num:", cache_num, "percent:", ratio)
        
        msg_list = []
        # cache_list = []

        remote_start = time.time()
        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            msg_list.append(remote_msg)
            # cache_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)
            # print("[recv msg]", idx)
        remote_time = time.time()-remote_start

        merge_start = time.time()
        # sort msg by server id and merge tensor together
        if len(msg_list) > 0:
            msg_list.sort(key=self._takeId)
            cache_id_tensor = F.cat(seq=[msg.id for msg in msg_list], dim=0)
            cache_data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)
            data_tensor = cache_data_tensor[:, :-1]
            state_sum_tensor = cache_data_tensor[:, -1]
            # 更新本地缓存
            local_id = cache_id_tensor
            if (name+'-g2l-' in self._has_data) == True:
                local_id = self._data_store[name+'-g2l-'][cache_id_tensor] # global id -> local id的映射
            # lock.acquire()
            if (name+'-l2p-' in self._has_data) == True:
                local_id = self._data_store[name+'-l2p-'][local_id] # local id -> position id的映射     
            self._update_cache(name, local_id, state_sum_tensor, data_tensor, self._data_store)
            # lock.release()


    def set_clr(self, learning_rate):
        """Set learning rate
        """
        self.clr = learning_rate


    def set_local2global(self, l2g):
        self._l2g = l2g


    def get_local2global(self):
        return self._l2g

    def set_entity_num(self, entity_num):
        self.entity_num = entity_num


def connect_to_kvstore(args, entity_pb, relation_pb, l2g):
    """Create kvclient and connect to kvstore service
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    # print(l2g)
    my_client = KGEClient(server_namebook=server_namebook)

    my_client.set_clr(args.lr)
    # print("start connect")
    my_client.connect()
    # print("finish connect")
    if my_client.get_id() % args.num_client == 0:
        my_client.set_partition_book(name='entity_emb', partition_book=entity_pb)
        my_client.set_partition_book(name='relation_emb', partition_book=relation_pb)
    else:
        my_client.set_partition_book(name='entity_emb')
        my_client.set_partition_book(name='relation_emb')

    my_client.set_local2global(l2g)
    # my_client.set_entity_num(len(l2g))

    return my_client


def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model


def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, entity_topk, relation_topk, valid_samplers=None, rank=0, testi=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)

    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0

    pull_time = 0
    local_pull = 0
    remote_pull = 0
    local_push = 0
    remote_push = 0

    ent_remote_ratio = 0.0
    rel_remote_ratio = 0.0    
    ent_hit_ratio = 0.0
    rel_hit_ratio = 0.0
    hit_count = 0

    ent_local_num = 0
    ent_remote_num = 0
    rel_local_num = 0
    rel_remote_num = 0
    
    sum_time = 0
    sum_communicate = 0
    sum_compute = 0
    push_step = args.push_step

    local_cache = None
    if push_step > 0:
        local_cache = True
    else:
        local_cache = False

    # input_file_dir = '/home/sicong/my_task/samplers/'
    # input_file = input_file_dir + str(rank) + '-' + str(testi) +'.bin'
    # train_list, labels = load_graphs(input_file)
    # train_sampler = iter(train_sampler)
    IO_time = time.time() - start
    pos_g = None
    neg_g = None

    entity_topk_emb = th.tensor([], dtype=th.float32)
    relation_topk_emb = th.tensor([], dtype=th.float32)

    prefetch_list = []
    prefetch_iter = []
    for step in range(0, args.max_step):
        
        synchronize = None
        if (push_step > 0) and (step % push_step == 0):
            synchronize = True
            hit_count += 1
            # client.set_switch_code("entity_emb", 1)
        else:
            synchronize = False

        start1 = time.time()
        if synchronize == True and args.dynamic_prefetch > 0:
            prefetch_list = []
            prefetch_iter = []
            entity_idlist = []
            relation_idlist = []
            entity_numlist = []
            relation_numlist = []
            for i in range(0, args.push_step):
                p_g, n_g = next(train_sampler)
                prefetch_list.append((p_g, n_g))

                with th.no_grad():
                    entity_id = F.cat(seq=[p_g.ndata['id'], n_g.ndata['id']], dim=0)
                    relation_id = p_g.edata['id']

                    # 去重操作
                    entity_id = F.tensor(np.unique(F.asnumpy(entity_id)))
                    relation_id = F.tensor(np.unique(F.asnumpy(relation_id)))

                    entity_idlist += entity_id.tolist()
                    relation_idlist += relation_id.tolist()
            prefetch_iter = iter(prefetch_list)
            
            entity_count = dict(Counter(entity_idlist))
            relation_count = dict(Counter(relation_idlist))
            entity_dict = sorted(entity_count.items(),key = lambda x:x[1],reverse = True)
            relation_dict = sorted(relation_count.items(),key = lambda x:x[1],reverse = True)

            entity_idlist = []
            relation_idlist = []
            
            for item in entity_dict:
                entity_idlist.append(item[0])
                entity_numlist.append(item[1])
            for item in relation_dict:
                relation_idlist.append(item[0])
                relation_numlist.append(item[1])

            entity_id = th.tensor(entity_idlist, dtype=th.long)
            relation_id = th.tensor(relation_idlist, dtype=th.long) 

            l2g = client.get_local2global()
            global_entity_id = l2g[entity_id] 

            entity_topk, relation_topk = client.cache_topk(entity_tensor=global_entity_id, relation_tensor=relation_id, entity_numlist=entity_numlist, relation_numlist=relation_numlist, topk=args.topk, ent_topk=args.ent_topk, rel_topk=args.rel_topk)
 
        # if step < args.pre_sample:
        #     pos_g, neg_g = next(backup_train_sampler)
        # else:
        if args.dynamic_prefetch > 0:
            pos_g, neg_g = next(prefetch_iter)
        else:
            pos_g, neg_g = next(train_sampler)
        
        sample_time += time.time() - start1
        model.entity_emb.step = step
        model.relation_emb.step = step

        # print("[step]", step)
        
        # print("[step cache sync]", step, local_cache, synchronize)

        # print("[step ", step, "] pull start")
        start1 = time.time()
        if client is not None:
            local1, remote1, ent_remote, rel_remote, ent_hit, rel_hit, e_l_n, e_r_n, r_l_n, r_r_n, ent_k_emb, rel_k_emb  = model.pull_model(client, 
                                                                                                                                pos_g, neg_g, entity_topk, relation_topk, 
                                                                                                                                entity_topk_emb, relation_topk_emb,
                                                                                                                                local_cache=local_cache, synchronize=synchronize)
            if synchronize == True:
                entity_topk_emb = ent_k_emb
                relation_topk_emb = rel_k_emb
            
            local_pull += local1
            remote_pull += remote1

            ent_local_num += e_l_n
            ent_remote_num += e_r_n
            rel_local_num += r_l_n
            rel_remote_num += r_r_n

            ent_remote_ratio += ent_remote
            rel_remote_ratio += rel_remote
            ent_hit_ratio += ent_hit
            rel_hit_ratio += rel_hit
            # print('[step{}] local pull {:.3f}, remote pull: {:.3f}'.format(step, local1, remote1))
        pull_time += time.time() - start1
        # print("[step ", step, "] pull finish")

        start1 = time.time()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        # new code
        # print("[step ", step, "] push start")
        if client is not None:
            local1, remote1, ent_hit, rel_hit = model.push_gradient(args, client, local_cache=local_cache, synchronize=synchronize)
            local_push += local1
            remote_push += remote1
            # if synchronize == True:
            #     ent_hit_ratio += ent_hit
            #     rel_hit_ratio += rel_hit
        else:
            model.update(gpu_id)
        update_time += time.time() - start1 
        logs.append(log)
        # print("[step ", step, "] push finish")

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
            (step + 1) % args.force_sync_interval == 0:
            barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    print('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                print('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                time.time() - start))
                # print('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                #     rank, sample_time, forward_time, backward_time, update_time))
                # sum_time += remote_pull+remote_push+sample_time+forward_time+backward_time+local_pull+local_push
                # sum_communicate += remote_pull+remote_push
                # sum_compute += sample_time+forward_time+backward_time+local_pull+local_push

                sum_time += pull_time+update_time+sample_time+forward_time+backward_time
                sum_communicate += pull_time+update_time
                sum_compute += sample_time+forward_time+backward_time

                print('[proc {}] sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, localpull: {:.3f}, localpush: {:.3f}, compute: {:.3f}, pull: {:.3f}, remotepull: {:.3f}, remotepush: {:.3f},  update: {:.3f}, communicate: {:.3f}, compute_ratio: {:.3f}, entity_remote_ratio: {:.3f}, relation_remote_ratio: {:.3f}, entity_hit_ratio: {:.3f}, relation_hit_ratio: {:.3f}, entity_local_num : {}, entity_remote_num : {:}, relation_local_num : {:}, relation_remote_num : {:} '.format(
                      rank, sample_time, forward_time, backward_time, local_pull, local_push, sum_compute, pull_time, remote_pull, remote_push, update_time, sum_communicate, sum_compute/sum_time, ent_remote_ratio/(step+1), rel_remote_ratio/(step+1), ent_hit_ratio/(step+1), rel_hit_ratio/(step+1), ent_local_num, ent_remote_num, rel_local_num, rel_remote_num))
                
                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0

                ent_local_num = 0
                ent_remote_num = 0
                rel_local_num = 0
                rel_remote_num = 0

                pull_time = 0
                local_pull = 0
                remote_pull= 0
                local_push = 0
                remote_push = 0
                start = time.time()

        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            test(args, model, valid_samplers, rank, mode='Valid')
            print('[proc {}]validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()

    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    if args.async_update:
        model.finish_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)

def test(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    with th.no_grad():
        logs = []
        for sampler in test_samplers:
            for pos_g, neg_g in sampler:
                model.forward_test(pos_g, neg_g, logs, gpu_id)

        metrics = {}
        if len(logs) > 0:
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs]) / len(logs)
        if queue is not None:
            queue.put(logs)
        else:
            for k, v in metrics.items():
                print('[{}]{} average {}: {}'.format(rank, mode, k, v))
    test_samplers[0] = test_samplers[0].reset()
    test_samplers[1] = test_samplers[1].reset()

@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, rank, rel_parts, cross_rels, barrier)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test', queue=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, rank, mode, queue)

@thread_wrapped_func
def cache_pull_proc(args, entity_pb, relation_pb, l2g, lock):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    
    client = connect_to_kvstore(args, entity_pb, relation_pb, l2g)
    # client.set_switch_code("relation_emb", client.get_id())
    for testi in range(args.test_num):
        client.barrier()
        while client.get_switch_code("entity_emb") < 0:
            time.sleep(0.05)
        count = 0
        while True:
            switch_code = client.get_switch_code("entity_emb")

            if switch_code == -1:
                break
            client.async_pull("entity_emb", lock)
            client.async_pull("relation_emb", lock)

            buffer_id = client.get_buffer_id()
            client.set_buffer_id(buffer_id ^ 1)
            count += 1
            # time.sleep(0.1)
            
        # print("[cache_pull_proc before barrier]", client.get_id())
        print('[async proc {}][Train] {} steps and pull {} times'.format(client.get_machine_id(), args.max_step, count))
        client.barrier()
        # print("[cache_pull_proc after barrier]", client.get_id())


@thread_wrapped_func
def dist_train_test(args, model, train_sampler, entity_id, relation_id, entity_numlist, relation_numlist, entity_pb, relation_pb, l2g, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    
    # print(sys.getsizeof(train_sampler), sys.getsizeof(entity_id), sys.getsizeof(relation_id))

    client = connect_to_kvstore(args, entity_pb, relation_pb, l2g)
    # 初始化缓存
    test_client_id = 0

    warm_start = time.time()
    entity_topk = th.tensor([], dtype=th.long)
    relation_topk = th.tensor([], dtype=th.long)
    local_client_id = client.get_id() % (args.num_client  - args.async_proc_num)
    # backup_train_sampler = []
    if (client is not None) and (args.push_step > 0) and args.dynamic_prefetch == 0:

        l2g = client.get_local2global()
        global_entity_id = l2g[entity_id]

        if args.topk > 0: 
            entity_topk, relation_topk = client.cache_topk(entity_tensor=global_entity_id, relation_tensor=relation_id, entity_numlist=entity_numlist, relation_numlist=relation_numlist, topk=args.topk, ent_topk=args.ent_topk, rel_topk=args.rel_topk)
        # if args.rel_topk > 0:
        #     relation_topk = client.cache_topk(name='relation_emb', id_tensor=relation_id, topk=args.rel_topk)
        print('Total cache warm time {:.3f} seconds'.format(time.time() - warm_start))
    # client.barrier()
    # else:
        
    for testi in range(args.test_num):

        # client.set_switch_code("entity_emb", 0)
        client.barrier()
        train_time_start = time.time()
        train(args, model, train_sampler, entity_topk, relation_topk, None, rank, testi, rel_parts, cross_rels, barrier, client)
        # client.set_switch_code("entity_emb", -1)
        total_train_time = time.time() - train_time_start  
        client.barrier()
        
        # Release the memory of local model
        # model = None
        
        if (client.get_machine_id() == 0) and (rank == 0): # pull full model from kvstore
            # Pull model from kvstore
            test_start = time.time()

            args.num_test_proc = args.num_client
            dataset_full = dataset = get_dataset(args.data_path, args.dataset, args.format, args.data_files)
            args.train = False
            args.valid = False
            args.test = True
            args.strict_rel_part = False
            args.soft_rel_part = False
            args.async_update = False

            args.eval_filter = not args.no_eval_filter
            if args.neg_deg_sample_eval:
                assert not args.eval_filter, "if negative sampling based on degree, we can't filter positive edges."

            print('Full data n_entities: ' + str(dataset_full.n_entities))
            print("Full data n_relations: " + str(dataset_full.n_relations))

            eval_dataset = EvalDataset(dataset_full, args)

            if args.neg_sample_size_eval < 0:
                args.neg_sample_size_eval = args.neg_sample_size = eval_dataset.g.number_of_nodes()
                args.batch_size_eval = get_compatible_batch_size(args.batch_size_eval, args.neg_sample_size_eval)

            model_test = load_model(None, args, dataset_full.n_entities, dataset_full.n_relations)

            print("Pull relation_emb ...")
            relation_id = F.arange(0, model_test.n_relations)
            relation_data, _, _ = client.pull(name='relation_emb', id_tensor=relation_id)
            relation_data = relation_data[:, :-1]
            model_test.relation_emb.emb[relation_id] = relation_data
    
            print("Pull entity_emb ... ")
            # split model into 100 small parts
            start = 0
            percent = 0
            entity_id = F.arange(0, model_test.n_entities)
            count = int(model_test.n_entities / 100)
            end = start + count
            while True:
                print("Pull model from kvstore: %d / 100 ..." % percent)
                if end >= model_test.n_entities:
                    end = -1
                tmp_id = entity_id[start:end]
                entity_data, _, _ = client.pull(name='entity_emb', id_tensor=tmp_id)
                entity_data = entity_data[:, :-1]
                model_test.entity_emb.emb[tmp_id] = entity_data
                if end == -1:
                    break
                start = end
                end += count
                percent += 1
        
            if not args.no_save_emb:
                print("save model to %s ..." % args.save_path)
                save_model(args, model_test)

            print('Total train time {:.3f} seconds'.format(total_train_time))

            if args.test:
                model_test.share_memory()

                test_sampler_tails = []
                test_sampler_heads = []
                for i in range(args.num_test_proc):
                    test_sampler_head = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                    args.neg_sample_size_eval,
                                                                    args.neg_sample_size_eval,
                                                                    args.eval_filter,
                                                                    mode='chunk-head',
                                                                    num_workers=args.num_workers,
                                                                    rank=i, ranks=args.num_test_proc)
                    test_sampler_tail = eval_dataset.create_sampler('test', args.batch_size_eval,
                                                                    args.neg_sample_size_eval,
                                                                    args.neg_sample_size_eval,
                                                                    args.eval_filter,
                                                                    mode='chunk-tail',
                                                                    num_workers=args.num_workers,
                                                                    rank=i, ranks=args.num_test_proc)
                    test_sampler_heads.append(test_sampler_head)
                    test_sampler_tails.append(test_sampler_tail)

                eval_dataset = None
                dataset_full = None

                print("Run test, test processes: %d" % args.num_test_proc)

                queue = mp.Queue(args.num_test_proc)
                procs = []
                for i in range(args.num_test_proc):
                    proc = mp.Process(target=test_mp, args=(args,
                                                            model_test,
                                                            [test_sampler_heads[i], test_sampler_tails[i]],
                                                            i,
                                                            'Test',
                                                            queue))
                    procs.append(proc)
                    proc.start()

                total_metrics = {}
                metrics = {}
                logs = []
                for i in range(args.num_test_proc):
                    log = queue.get()
                    logs = logs + log
                
                for metric in logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

                test_time = time.time() - test_start

                print("-------------- Test ", testi+1, " result --------------")
                for k, v in metrics.items():
                    print('Test average {} : {}'.format(k, v))
                print('Test time: ', test_time, 's')
                print("-----------------------------------------")

                for proc in procs:
                    proc.join()

    if (client.get_machine_id() == 0) and (rank == 0):
        client.shut_down() # shut down kvserver
