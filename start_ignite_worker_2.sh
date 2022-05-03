#!/bin/bash
python -u $1 --nproc_per_node 2 --backend gloo --nnodes=1 --node_rank=2 --master_addr=192.168.3.234 --master_port=2222