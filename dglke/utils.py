# -*- coding: utf-8 -*-
#
# utils.py
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

import math
import os
import argparse
import json

def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
            old_batch_size, neg_sample_size, batch_size))
    return batch_size

def save_model(args, model):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    model.save_emb(args.save_path, args.dataset)

    # We need to save the model configurations as well.
    conf_file = os.path.join(args.save_path, 'config.json')
    with open(conf_file, 'w') as outfile:
        json.dump({'dataset': args.dataset,
                   'model': args.model_name,
                   'emb_size': args.hidden_dim,
                   'max_train_step': args.max_step,
                   'batch_size': args.batch_size,
                   'neg_sample_size': args.neg_sample_size,
                   'lr': args.lr,
                   'gamma': args.gamma,
                   'double_ent': args.double_ent,
                   'double_rel': args.double_rel,
                   'neg_adversarial_sampling': args.neg_adversarial_sampling,
                   'adversarial_temperature': args.adversarial_temperature,
                   'regularization_coef': args.regularization_coef,
                   'regularization_norm': args.regularization_norm},
                   outfile, indent=4)

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge '\
                                  'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '\
                                  'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs,'\
                                  'the foramt should be built_in. For users own knowledge graphs,'\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE'\
                                  'on their own datasets. If the format is raw_udd_{htr},'\
                                  'users need to provide train_file [valid_file] [test_file].'\
                                  'If the format is udd_{htr}, users need to provide'\
                                  'entity_file relation_file train_file [valid_file] [test_file].'\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--save_path', type=str, default='ckpts',
                          help='the path of the directory where models and logs are saved.')
        self.add_argument('--no_save_emb', action='store_true',
                          help='Disable saving the embeddings under save_path.')
        self.add_argument('--max_step', type=int, default=80000,
                          help='The maximal number of steps to train the model.'\
                                  'A step trains the model with a batch of data.')
        self.add_argument('--batch_size', type=int, default=1024,
                          help='The batch size for training.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for validation and test.')
        self.add_argument('--neg_sample_size', type=int, default=256,
                          help='The number of negative samples we use for each positive sample in the training.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the training.'\
                                  'When this option is turned on, the number of negative samples per positive edge'\
                                  'will be doubled. Half of the negative samples are generated uniformly while'\
                                  'the other half are generated proportional to vertex degree.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The number of negative samples we use to evaluate a positive sample.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='Print runtime of different components every x steps.')
        self.add_argument('--eval_interval', type=int, default=10000,
                          help='Print evaluation results on the validation dataset every x steps'\
                                  'if validation is turned on')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the test set after the model is trained.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to train the model in parallel.'\
                                  'In multi-GPU training, the number of processes by default is set to match the number of GPUs.'\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process.'\
                                  'This argument is used for multiprocessing training.')
        self.add_argument('--force_sync_interval', type=int, default=-1,
                          help='We force a synchronization between processes every x steps for'\
                                  'multiprocessing training. This potentially stablizes the training process'
                                  'to get a better performance. For multiprocessing training, it is set to 1000 by default.')
        self.add_argument('--hidden_dim', type=int, default=400,
                          help='The embedding size of relation and entity')
        self.add_argument('--lr', type=float, default=0.01,
                          help='The learning rate. DGL-KE uses Adagrad to optimize the model parameters.')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entitiy dim for complex number It is used by RotatE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number.')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='Indicate whether to use negative adversarial sampling.'\
                                  'It will weight negative samples with higher scores more.')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                          help='The temperature used for negative adversarial sampling.')
        self.add_argument('-rc', '--regularization_coef', type=float, default=0.000002,
                          help='The coefficient for regularization.')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization.')

        # new code
        self.add_argument('--push_step', type=int, default=0,
                          help='How many iterations the worker push parameters')
        self.add_argument('--test_num', type=int, default=1,
                          help='The number of tests')
        self.add_argument('--ent_topk', type=int, default=-1,
                          help='The cache ratio of entity embeddings, default = 0')
        self.add_argument('--rel_topk', type=int, default=-1,
                          help='The cache ratio of relation embeddings, default = 0')
        self.add_argument('--topk', type=int, default=0,
                          help='The cache ratio of whole embeddings, default = 0')
        self.add_argument('--async_proc_num', type=int, default=0,
                          help='Using async proc for cache updating, default 0')
        self.add_argument('--pre_sample', type=int, default=0,
                          help='Using for pre sampling and caching, default 0')
        self.add_argument('--dynamic_prefetch', type=int, default=0,
                          help='Using for dynamic prefetch, default 0')


class RWLock(object):


    def __init__(self):
        self.lock = threading.Lock()
        self.rcond = threading.Condition(self.lock)
        self.wcond = threading.Condition(self.lock)
        self.read_waiter = 0    # 等待获取读锁的线程数
        self.write_waiter = 0   # 等待获取写锁的线程数
        self.state = 0          # 正数：表示正在读操作的线程数   负数：表示正在写操作的线程数（最多-1）
        self.owners = []        # 正在操作的线程id集合
        self.write_first = True # 默认写优先，False表示读优先

    def write_acquire(self, blocking=True):
        # 获取写锁只有当
        me = threading.get_ident()
        with self.lock:
            while not self._write_acquire(me):
                if not blocking:
                    return False
                self.write_waiter += 1
                self.wcond.wait()
                self.write_waiter -= 1
        return True

    def _write_acquire(self, me):
        # 获取写锁只有当锁没人占用，或者当前线程已经占用
        if self.state == 0 or (self.state < 0 and me in self.owners):
            self.state -= 1
            self.owners.append(me)
            return True
        if self.state > 0 and me in self.owners:
            raise RuntimeError('cannot recursively wrlock a rdlocked lock')
        return False

    def read_acquire(self, blocking=True):
        me = threading.get_ident()
        with self.lock:
            while not self._read_acquire(me):
                if not blocking:
                    return False
                self.read_waiter += 1
                self.rcond.wait()
                self.read_waiter -= 1
        return True

    def _read_acquire(self, me):
        if self.state < 0:
            # 如果锁被写锁占用
            return False

        if not self.write_waiter:
            ok = True
        else:
            ok = me in self.owners
        if ok or not self.write_first:
            self.state += 1
            self.owners.append(me)
            return True
        return False

    def unlock(self):
        me = threading.get_ident()
        with self.lock:
            try:
                self.owners.remove(me)
            except ValueError:
                raise RuntimeError('cannot release un-acquired lock')

            if self.state > 0:
                self.state -= 1
            else:
                self.state += 1
            if not self.state:
                if self.write_waiter and self.write_first:   # 如果有写操作在等待（默认写优先）
                    self.wcond.notify()
                elif self.read_waiter:
                    self.rcond.notify_all()
                elif self.write_waiter:
                    self.wcond.notify()

    read_release = unlock
    write_release = unlock

def mergesort(list1, list2):
    i = 0
    j = 0
    k = 0
    index1 = []
    index2 = []
    result = []
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            index1.append(k)
            i += 1
        elif list1[i] > list2[j]:
            result.append(list2[j])
            index2.append(k)
            j += 1
        else:
            result.append(list1[i])
            index1.append(k)
            index2.append(k)
            i += 1
            j += 1
        k += 1
    while i < len(list1):
        result.append(list1[i])
        index1.append(k)
        i += 1
        k += 1
    while j < len(list2):
        result.append(list2[j])
        index2.append(k)
        j += 1
        k += 1
    return result, index1, index2

def merge_intersection(list1, list2):
    i = 0
    j = 0
    k = 0
    index1 = []
    index2 = []
    index3 = []
    result = []
    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            result.append(list1[i])
            index2.append(i)
            i += 1
        elif list1[i] > list2[j]:
            j += 1
        else:
            index1.append(i)
            index3.append(j)
            i += 1
            j += 1
            k += 1
    while i < len(list1):
        result.append(list1[i])
        index2.append(i)
        i += 1
    return result, index2, index1, index3
