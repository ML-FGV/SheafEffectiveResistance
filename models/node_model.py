import torch
import torch.nn as nn
from torch.nn import ModuleList, Dropout, ReLU
from torch_geometric.nn import GCNConv, RGCNConv, SAGEConv, GINConv, FiLMConv, global_mean_pool
from models.sheaf_model import FlatBundleConv, FlatGenSheafConv

class RGINConv(torch.nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGINConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.self_loop_conv = torch.nn.Linear(in_features, out_features)
        convs = []
        for i in range(self.num_relations):
            convs.append(GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features))))
        self.convs = ModuleList(convs)
    def forward(self, x, edge_index, edge_type):
        x_new = self.self_loop_conv(x)
        for i, conv in enumerate(self.convs):
            rel_edge_index = edge_index[:, edge_type==i]
            x_new += conv(x, rel_edge_index)
        return x_new

class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_relations = args.num_relations
        self.layer_type = args.layer_type
        num_features = [args.input_dim] + list(args.hidden_layers) + [args.output_dim]
        self.num_layers = len(num_features) - 1
        layers = []
        if "Sheaf" not in self.layer_type:
            for i, (in_features, out_features) in enumerate(zip(num_features[:-1], num_features[1:])):
                layers.append(self.get_layer(in_features, out_features))
        else:
            self.sheaf_emb = nn.Linear(num_features[0], num_features[1]*args.d)
            for i in range(self.num_layers - 2):
                layers.append(self.get_layer(args.hidden_dim, args.hidden_dim))
            self.sheaf_readout = torch.nn.Linear(self.args.hidden_dim * self.args.d, self.args.output_dim)
        self.layers = ModuleList(layers)

        self.reg_params = list(layers[0].parameters())
        self.non_reg_params = list([p for l in layers[1:] for p in l.parameters()])

        self.dropout = Dropout(p=args.dropout)
        self.act_fn = ReLU()

    def get_layer(self, in_features, out_features):
        if self.layer_type == "GCN":
            return GCNConv(in_features, out_features)
        elif self.layer_type == "R-GCN":
            return RGCNConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "GIN":
            return GINConv(nn.Sequential(nn.Linear(in_features, out_features),nn.BatchNorm1d(out_features), nn.ReLU(),nn.Linear(out_features, out_features)))
        elif self.layer_type == "R-GIN":
            return RGINConv(in_features, out_features, self.num_relations)
        elif self.layer_type == "SAGE":
            return SAGEConv(in_features, out_features)
        elif self.layer_type == "FiLM":
            return FiLMConv(in_features, out_features)
        elif self.layer_type == "Sheaf":
            return FlatBundleConv(in_channels=in_features,
                                  out_channels=out_features,
                                  stalk_dimension=self.args.d,
                                  dropout=self.args.dropout,
                                  linear_emb=self.args.linear_emb,
                                  gnn_type=self.args.gnn_type,
                                  gnn_layers=self.args.gnn_layers,
                                  gnn_hidden=self.args.hidden_dim)
        elif self.layer_type == "GenSheaf":
            return FlatGenSheafConv(in_channels=in_features,
                                  out_channels=out_features,
                                  stalk_dimension=self.args.d,
                                  dropout=self.args.dropout,
                                  linear_emb=self.args.linear_emb,
                                  gnn_type=self.args.gnn_type,
                                  gnn_layers=self.args.gnn_layers,
                                  gnn_hidden=self.args.hidden_dim)
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        reff_per_layer = torch.zeros((self.num_layers,), device=x.device)
        if "Sheaf" in self.layer_type:
            x = self.sheaf_emb(x)
            x = torch.nn.functional.gelu(x)
        for i, layer in enumerate(self.layers):
            reff = False
            if self.layer_type in ["R-GCN", "R-GIN"]:
                x = layer(x, edge_index, edge_type=graph.edge_type)
            elif "Sheaf" in self.layer_type:
                x, reff_per_layer[i] = layer(x, edge_index, graph, reff=reff)
            else:
                x = layer(x, edge_index)
            if i != self.num_layers - 1:
                x = self.act_fn(x)
                x = self.dropout(x)
        if "Sheaf" in self.layer_type:
            x = self.sheaf_readout(x)
        return x
