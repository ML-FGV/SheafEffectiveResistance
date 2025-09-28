from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr

from tqdm import tqdm
from torch_geometric.utils import get_laplacian
from scipy.linalg import pinv

import wandb

largest_cc = LargestConnectedComponents()


cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
squirrel = WikipediaNetwork(root="data", name="squirrel")
actor = Actor(root="data")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, "chameleon": chameleon, "squirrel": squirrel, "actor": actor, "cora": cora, "citeseer": citeseer, "pubmed": pubmed}
#datasets = {"cornell": cornell, "wisconsin": wisconsin}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "d": 5,
    "gnn_type": "GIN",
    "gnn_layers": 1,
    "linear_emb": False,
    "dropout": 0.5,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "R-GCN",
    "display": False,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "fosr",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 100,
    "dataset": None
    })

results = []
args = default_args
args += get_args_from_input()

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    wandb.init(project="SheafEffectiveResistance", config=args, allow_val_change=True)
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    print(f"TESTING: {key} ({default_args.rewiring}) with {args.layer_type} model")
    dataset = datasets[key]
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
    elif args.rewiring == "sdrf":
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, is_undirected=True)
    #print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
    for trial in range(args.num_trials):
        #print(f"TRIAL {trial+1}")
        print(args)
        train_acc, validation_acc, test_acc = Experiment(args=wandb.config, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
    
    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    train_ci = 200 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 200 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 200 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    # log_to_file(f"RESULTS FOR {key} ({default_args.rewiring}):\n")
    # log_to_file(f"average acc: {np.mean(accuracies)}\n")
    # log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    
    print(f"RESULTS FOR {key} ({default_args.rewiring}):\n")
    print(f"average acc: {test_mean}\n")
    print(f"plus/minus:  {test_ci}\n\n")
    wandb.log({"avg_val_acc": val_mean, "avg_test_acc": test_mean, "test_acc_std": test_ci})
    wandb.finish()
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "avg_accuracy": test_mean,
        "ci":  test_ci
        })
    # results_df = pd.DataFrame(results)
    # with open('results/node_classification.csv', 'a') as f:
    #     results_df.to_csv(f, mode='a', header=f.tell()==0)
