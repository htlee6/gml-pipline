# Performance Analysis of Prediction on Molecular Graphs with Graph Neural Networks

## Install Python Packages

``` bash
pip install numpy
pip install torch==1.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip install rdkit-pypi
pip install ogb
pip install pytorch-ignite
pip install dgl==0.6.1 dglgo -f https://data.dgl.ai/wheels/repo.html
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

**P.S.**

 There is a small known issue w.r.t the PyG setup. To run this script, we need to modify a line of the source code, including
1) In `path/to/python/site-packages/ogb/graphproppred/__init__.py`
```python
# from .evaluate import Evaluator # comment this line
from .dataset import GraphPropPredDataset
```
2) When this task is finished, **REVERT THE SOURCE CODE**

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
