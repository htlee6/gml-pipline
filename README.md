# gml-pipline

## Install Python Packages

``` bash
pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Run the Scripts
### PyG
Run with default params 
```bash
python pyg.py
```
Run with custom params
```bash
python pyg.py --gnn gcn --batch_size 32 --epochs 20
```

### DGL (PyTorch backend)
Run with default params
```bash
python dgl_torch.py
```
Run with custom params
```bash
python dgl_torch.py --gnn gcn --batch_size 32 --epochs 20
```

### PyG+Ignite (Distributed)
Say your cluster contains `n=3` hosts
```bash
n=3
N_PROC="YOUR_N_PROC"
MASTER_ADDR="YOUR_MASTER_ADDR"
MASTER_PORT="YOUR_MASTER_PORT"

# on host 1
python -u ignite_pyg_t.py --nproc_per_node=$N_PROC --backend gloo --nnodes=$n --node_rank=0 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT

# on host 2
python -u ignite_pyg_t.py --nproc_per_node=$N_PROC --backend gloo --nnodes=$n --node_rank=1 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT

# on host 3
python -u ignite_pyg_t.py --nproc_per_node=$N_PROC --backend gloo --nnodes=$n --node_rank=2 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT

```