# HET-KG

HET-KG is implemented on top of DGL-KE and supports hot embedding cache optimization to reduce training time.

## Installation
```
$ yum install python3
$ pip3 install torch dgl==0.4.3 dglke==0.1.0
$ sudo cp dglke /usr/local/lib/python3.6/site-packages/ (your installation directory)
```

## Start

```
Based on DGL-KE, HET-KG adds parameters related to hot embedding training.

dglke_dist_train --path ~/my_task  --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model_name TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 16.0 --lr 0.25 --batch_size 32 --neg_sample_size 8 --max_step 10000   --log_interval 10000    \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --num_thread 1 --push_step 16 --topk 64 --dynamic_prefetch 1

explanation:
'--push_step x' synchronize hot embeddings with parameter servers every x rounds
'--topk y' number of hot embeddings in each worker (if the 'ent_topk' and 'rel_topk' are not set, the hot embeddings are selected according to the frequency)
'--ent_topk' number of entity hot embeddings in each worker
'--rel_topk' number of relation hot embeddings in each worker
'--dynamic_prefetch z' z=0 (using constant partial stale), z=1 (using dynamic partial stale)
```

## Citation

* Sicong Dong, Xupeng Miao, Pengkai Liu, Xin Wang, Bin Cui, Jianxin Li. HET-KG: Communication-Efficient Knowledge Graph Embedding Training via Hotness-Aware Cache. The 38th IEEE International Conference on Data Engineering (ICDE 2022, Research Track).
