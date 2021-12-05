# HotKE
## Installation
```
$ yum install python3
$ pip3 install torch dgl==0.4.3 dglke==0.1.0
$ sudo cp dglke /usr/local/lib/python3.6/site-packages/
```

## Start

```
dglke_dist_train --path ~/my_task  --ip_config ~/my_task/ip_config.txt \
--num_client_proc 16 --model_name TransE_l2 --dataset FB15k --data_path ~/my_task --hidden_dim 400 \
--gamma 16.0 --lr 0.25 --batch_size 32 --neg_sample_size 8 --max_step 10000   --log_interval 10000    \
--batch_size_eval 16 --test -adv --regularization_coef 1.00E-07 --num_thread 1 --push_step 16 --topk 64 --dynamic_prefetch 1


explanation:
'--push_step x' synchronize hot embeddings with parameter servers every x rounds;
'--topk y' number of hot embeddings in each worker;
'--dynamic_prefetch z' z=0 (using constant partial stale), z=1 (using dynamic partial stale)
```
