import argparse
from ast import arg
from cgi import test

import ignite.distributed as idist
from numpy import dtype
from setuptools import setup
import torch
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy
from ignite.contrib.metrics import ROC_AUC
from torch.nn import NLLLoss
from torch.optim import SGD, Adam
from ignite.utils import setup_logger
from torch.utils.data import Dataset
import torch.nn.functional as F

# rom torch_geometric.datasets import Planetoid
import ogb
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

from time import time

from tqdm import tqdm

# torch.set_default_dtype(torch.float64)

dataset_outside = PygGraphPropPredDataset(name="ogbg-molhiv")
data_0_outside = dataset_outside[0]

# Use Collater() as `collate_fn`, code from pyg source

from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from torch_geometric.data import Data, HeteroData, Dataset, Batch

class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data) or isinstance(elem, HeteroData):
            return Batch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)

# Custmoize your model

from gnn import GNN

def training(rank, config):

    import torch.distributed as dist
    assert dist.is_available() and dist.is_initialized()

    # assert dist.get_world_size() == 

    # Specific ignite.distributed
    print(
        idist.get_rank(),
        ": run with config:",
        config,
        "- backend=",
        idist.backend(),
        "- world size",
        idist.get_world_size(),
    )

    device = idist.device()
    # Data preparation:
    # dataset = RndDataset(nb_samples=config["nb_samples"])
    # new_dataset = torch.utils.data.Dataset(dataset)

    # Specific ignite.distributed
    # train_loader = idist.auto_dataloader(dataset, batch_size=config["batch_size"], collate_fn=Collater(False, []))
    # print(config['dataset'])
    dataset = PygGraphPropPredDataset(name=config['dataset'])
    
    split_idx = dataset.get_idx_split()
    # print()
    train_loader = idist.auto_dataloader(dataset[split_idx['train']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))
    valid_loader = idist.auto_dataloader(dataset[split_idx['valid']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))
    test_loader = idist.auto_dataloader(dataset[split_idx['test']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))

    # Model, criterion, optimizer setup
    if config['gnn'] == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = config['num_layer'], emb_dim = config['emb_dim'], drop_ratio = config['drop_ratio'], virtual_node = False).to(device)
    elif config['gnn'] == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = config['num_layer'], emb_dim = config['emb_dim'], drop_ratio = config['drop_ratio'], virtual_node = True).to(device)
    elif config['gnn'] == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = config['num_layer'], emb_dim = config['emb_dim'], drop_ratio = config['drop_ratio'], virtual_node = False).to(device)
    elif config['gnn'] == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = config['num_layer'], emb_dim = config['emb_dim'], drop_ratio = config['drop_ratio'], virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')
   
   
    # print("Preparing Model")
    model = idist.auto_model(model)
    # print("Preparing Opt")
    optimizer = idist.auto_optim(Adam(model.parameters(), lr=0.01))
    if dataset.task_type == 'classification':
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.MSELoss()

    # Training loop log param
    log_interval = config["log_interval"]
    '''
    def _train_step(engine, batch):

        data = batch
        data = data.to(device)
        target = batch.y.to(device)

        # print(data.dtype, target.dtype)

        optimizer.zero_grad()
        output = model(data)
        # Add a softmax layer
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # print(probabilities.shape, target.squeeze(1).shape)
        loss_val = criterion(probabilities.to(torch.float64), target.squeeze(1))
        loss_val.backward()
        optimizer.step()

        return loss_val
    '''

    def _train(engine, foo):
        model.train()
        for step, batch in enumerate(train_loader):
            # if step % 3 == 0:
            #     print("current step " + str(step))
            batch = batch.to(device)

            if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                pass
            else:
                pred = model(batch)
                optimizer.zero_grad()
                ## ignore nan targets (unlabeled) when computing training loss.
                is_labeled = batch.y == batch.y
                loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                loss.backward()
                optimizer.step()
    
    # torch.no_grad()
    # return_dict = {"y_pred": model(dataset[split_idx['test']]), "y": dataset[split_idx['test']].y}

    # Running the _train_step function on whole batch_data iterable only once
    # trainer = Engine(_train_step)
    # trainer = Engine(train, {"model": model, "device": device, "loader": train_loader, "optimizer": optimizer})
    # print("About to Start")
    trainer = Engine(_train)

    metrics_dict = {"ROC_AUC": ROC_AUC(), "Acc": Accuracy()}
    # metrics_dict = {"Acc": Accuracy()}
    def graph_prepare_batch(batch, **kwargs):
        class Features():
            def __init__(self):
                pass
        f = Features()
        f.x, f.edge_index, f.edge_attr, f.batch, y = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.y
        return f, y
    
    def custom_output_transform(x, y, y_pred):
        return ((y_pred > 0).int(), y)

    evaluator = create_supervised_evaluator(model=model, metrics=metrics_dict, prepare_batch=graph_prepare_batch, output_transform=custom_output_transform)
    evaluator.logger = setup_logger("evaluator")

    pbar = tqdm(initial=0, leave=False, total=len(train_loader), desc=f"Iteration - loss: {0: .2f}")
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_progress_bar(engine):
        pbar.update(log_interval)
    
    # Add a logger
    @trainer.on(Events.ITERATION_COMPLETED(every=100))
    def log_training(engine):
        '''
        print(
            "Process {}/{} Train Epoch: {} [{}/{}]\tLoss: {}".format(
                idist.get_rank(),
                idist.get_world_size(),
                trainer.state.epoch,
                trainer.state.iteration * len(trainer.state.batch[0]),
                len(dataset) / idist.get_world_size(),
                trainer.state.output,
            )
        
        )
        '''
        batch_loss = engine.state.output
        lr = optimizer.param_groups[0]['lr']
        e = engine.state.epoch
        n = engine.state.max_epochs
        i = engine.state.iteration
        print(f"Epoch {e}/{n} : {i} - batch loss: {batch_loss}, lr: {lr}")

    @trainer.on(Events.COMPLETED)
    def eval_testloader(engine):

        try:
            evaluator.run(test_loader)
        except:
            print(evaluator.state.output)
        
        metrics = evaluator.state.metrics
        if 'Acc' in metrics.keys():
            avg_acc = metrics['Acc']
            tqdm.write('Average Acc: ' + str(avg_acc))
        if 'ROC_AUC' in metrics.keys():
            avg_roc_auc = metrics['ROC_AUC']
            tqdm.write('Average ROC_AUC ' + str(avg_roc_auc))

    
    state = trainer.run(train_loader, max_epochs=1)
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytorch Ignite - idist")
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--nb_samples", type=int, default=128)
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--master_addr", type=str)
    parser.add_argument("--master_port", type=int)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')

    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
 
    args_parsed = parser.parse_args()

    device = torch.device("cuda:" + str(args_parsed.device)) if torch.cuda.is_available() else torch.device("cpu")

    # idist from ignite handles multiple backend (gloo, nccl, horovod, xla)
    # and launcher (torch.distributednch, horovodrun, slurm)
    config = {
        "log_interval": args_parsed.log_interval,
        "batch_size": args_parsed.batch_size,
        "nb_samples": args_parsed.nb_samples,
        "gnn": args_parsed.gnn,
        "dataset": args_parsed.dataset,
        "num_layer": args_parsed.num_layer,
        "emb_dim": args_parsed.emb_dim,
        "drop_ratio": args_parsed.drop_ratio
    }

    # Dataloader Prepare
    dataset = PygGraphPropPredDataset(name=config['dataset'])
    
    '''
    split_idx = dataset.get_idx_split()
    # print()
    train_loader = idist.auto_dataloader(dataset[split_idx['train']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))
    valid_loader = idist.auto_dataloader(dataset[split_idx['valid']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))
    test_loader = idist.auto_dataloader(dataset[split_idx['test']], batch_size=config['batch_size'],shuffle=True, collate_fn=Collater(False, []))
    print(len(split_idx['train']))
    '''

    # Process spwan config # 
    spawn_kwargs = dict()
    spawn_kwargs["nproc_per_node"] = args_parsed.nproc_per_node
    spawn_kwargs["nnodes"] = args_parsed.nnodes
    spawn_kwargs["node_rank"] = args_parsed.node_rank
    spawn_kwargs["master_addr"] = args_parsed.master_addr
    spawn_kwargs["master_port"] = args_parsed.master_port

    # Specific ignite.distributed
    start = time()
    with idist.Parallel(backend=args_parsed.backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)
        # parallel.run(evaluation, test_loader, config)

    end = time()
    print('training time: ' + str(end-start))