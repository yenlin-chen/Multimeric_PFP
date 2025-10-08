if __name__ == '__main__':
    # from check_model_args import check_args
    from mGCNConv import mGCNConv
else:
    # from .check_model_args import check_args
    from .mGCNConv import mGCNConv

import json
import sys
from os import path

import torch
from torch import nn
from torchinfo import summary

import torch_geometric as pyg

model_version = 'v1'

def simple_embedding_block(
        neuron_ls,
        dropout_rate,
        activation='leakyrelu'
    ):

    ## BUILD SEQUENTIAL MODEL
    mods = []
    for layer_idx in range(len(neuron_ls) - 1):
        dim_input = neuron_ls[layer_idx]
        dim_output = neuron_ls[layer_idx + 1]

        # linear connection
        mods.append(nn.Linear(dim_input, dim_output))

        # dropout
        mods.append(nn.Dropout(p=dropout_rate))

        # activation
        if activation == 'leakyrelu':
            mods.append(nn.LeakyReLU())
        elif activation == 'selu':
            mods.append(nn.SELU())

    return nn.Sequential(*mods)

class SimplifiedMultiGCN(nn.Module):
    def __init__(
        self,
        dim_model_output,

        # FEATURE SELECTION
        use_pi=False,

        # GRAPH CONVOLUTION SETUP
        node_feat_name=None,
        node_feat_size=None,
        gnn_type=None,
        gat_atten_heads=None,
        dim_node_hidden_dict=None,
        conv_norm=None,
        norm_graph_input=None,
        norm_graph_output=None,
        graph_global_pool=None,
        graph_dropout_rate=None,
        dropfeat_rate=None,
        dropedge_rate=None,
        dropnode_rate=None,
        jk_concat=None,

        # PERSISTENCE IMAGES SETUP
        pi_dropout_rate=None,
        dim_pi_embedding=None,

        # FC SETUP
        fc_hidden_ls=None,
        n_fc_hidden_layers=None,
        fc_norm=True,
        norm_fc_input=False,
        norm_fc_output=False,
        fc_dropout_rate=0.5,

        # OTHERS
        debug=False
    ):

        if debug:
            torch.autograd.set_detect_anomaly(True)

        ################################################################
        # SAVE A COPY OF ARGUMENTS PASSED
        ################################################################

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        # arguments required in forward()
        self.graph_dims = dim_node_hidden_dict.keys()
        self.use_pi = use_pi
        self.dropedge_rate = dropedge_rate
        self.dropnode_rate = dropnode_rate
        self.node_feat_name = node_feat_name

        # internal parameters
        self.training = True
        self.debug = debug

        super().__init__()

        ################################################################
        # INSTANTIATE POOLING LAYERS
        ################################################################
        if graph_global_pool is None:
            self.graph_pool = None
        elif graph_global_pool == 'mean':
            self.graph_pool = pyg.nn.global_mean_pool
        elif graph_global_pool == 'max':
            self.graph_pool = pyg.nn.global_max_pool
        elif graph_global_pool == 'sum':
            self.graph_pool = pyg.nn.global_add_pool
        else:
            raise ValueError(
                f'`graph_global_pool` must be "mean", "max", or "sum", '
                f'not {graph_global_pool}'
            )

        ################################################################
        # KEEP TRACK OF THE NUMBER OF FEATURE FOR FC LAYERS
        ################################################################

        dim_fc_input = 0

        ################################################################
        # INSTANTIATE GRAPH CONVOLUTIONAL LAYERS
        ################################################################

        n_dims = len(self.graph_dims)

        assert n_dims != 0
        assert len(dim_node_hidden_dict) == n_dims

        self.conv_block_dict = nn.ModuleDict({})
        for dim_name in self.graph_dims:

            mods = []

            dim_node_ls = [ node_feat_size ] + dim_node_hidden_dict[dim_name]
            dim_node_hidden_ls = dim_node_ls[1:]
            dim_node_hidden = dim_node_ls[-1]

            total_node_dims = 0

            for layer_idx in range(len(dim_node_hidden_ls)):
                dim_input = dim_node_ls[layer_idx]
                dim_output = dim_node_ls[layer_idx + 1]

                next_idx = layer_idx

                # only for the first layer
                if layer_idx == 0:
                    # dropfeat
                    if dropfeat_rate:
                        mods.append((
                            nn.Dropout(p=dropfeat_rate),
                            f'x{next_idx} -> x{layer_idx+1}'
                        ))
                        next_idx = layer_idx+1

                # exclude the first layer
                if layer_idx != 0 or norm_graph_input:
                    # normalization
                    if conv_norm:
                        mods.append((
                            pyg.nn.GraphNorm(dim_input),
                            f'x{next_idx}, batch -> x{layer_idx+1}'
                        ))
                        next_idx = layer_idx+1

                # convolution
                if gnn_type == 'gcn':
                    conv = pyg.nn.GCNConv(
                        dim_input, dim_output,
                        add_self_loops=True, bias=True
                    )
                elif gnn_type == 'gin':
                    intermediate = (dim_input+dim_output) // 2
                    gin_nn = nn.Sequential(
                        nn.Linear(dim_input, intermediate),
                        nn.BatchNorm1d(intermediate),
                        nn.ReLU(),
                        nn.Linear(intermediate, dim_output),
                        nn.ReLU()
                    )
                    conv = pyg.nn.GINConv(
                        nn=gin_nn,
                        train_eps=True
                    )
                elif gnn_type == 'gat':
                    assert dim_output % gat_atten_heads == 0
                    conv = pyg.nn.GATConv(
                        dim_input, dim_output//gat_atten_heads,
                        heads=gat_atten_heads, dropout=graph_dropout_rate,
                        add_self_loops=True
                    )
                else:
                    raise ValueError(
                        f'`gnn_type` must be "gcn", "gin", or "gat", '
                        f'not "{gnn_type}"'
                    )
                mods.append((
                    conv,
                    f'x{next_idx}, edge_index -> x{layer_idx+1}'
                ))

                # dropout
                if graph_dropout_rate:
                    mods.append((
                        nn.Dropout(p=graph_dropout_rate),
                        f'x{layer_idx+1} -> x{layer_idx+1}'
                    ))

                # activation
                mods.append((
                    nn.LeakyReLU(),
                    f'x{layer_idx+1} -> x{layer_idx+1}'
                ))

            feats = [f'x{i+1}' for i in range(len(dim_node_hidden_ls))]

            if jk_concat:
                # jumping knowledge connections
                mods.append((lambda *x: [*x], ', '.join(feats)+' -> xs'))

                mods.append((pyg.nn.JumpingKnowledge('cat'), 'xs -> x'))
                graph_embedding_size = sum(dim_node_hidden_ls)

                # normalization for concatenated node embeddings
                if norm_graph_output:
                    raise NotImplementedError
                    mods.append(
                        (pyg.nn.GraphNorm(graph_embedding_size),
                         'x, batch -> x')
                    )

                # sum size of embeddings accross all layers and graph dims
                total_node_dims += graph_embedding_size

            else:

                # normalization for node embeddings
                if norm_graph_output:
                    raise NotImplementedError
                    mods.append(
                        (pyg.nn.GraphNorm(dim_node_hidden),
                         f'x{feats[-1]}, batch -> x{feats[-1]}')
                    )

                # no jumping knowledge connections
                total_node_dims += dim_node_hidden

            # for m in mods:
            #     print(m)
            self.conv_block_dict[dim_name] = (
                pyg.nn.Sequential('x0, edge_index, batch', mods)
            )

            dim_fc_input += total_node_dims

        ################################################################
        # INSTANTIATE GRAPH CONVOLUTIONAL LAYERS
        ################################################################
        if use_pi:
            self.pi_block = nn.Sequential(
                nn.Linear(625, dim_pi_embedding),
                nn.LayerNorm(dim_pi_embedding),
                nn.Dropout(p=pi_dropout_rate),
                nn.LeakyReLU(),
            )
            dim_fc_input += dim_pi_embedding

        ################################################################
        # INSTANTIATE FULLY CONNECTED LAYERS
        ################################################################

        if fc_hidden_ls is None:
            factor = (dim_fc_input-dim_model_output)/(n_fc_hidden_layers+1)
            if factor !=0:
                fc_hidden_ls = [
                    int(dim_model_output+factor*i)
                    for i in range(1,n_fc_hidden_layers+1)[::-1]
                ]
            else:
                fc_hidden_ls = [dim_model_output] * n_fc_hidden_layers

        fc_neuron_ls = [dim_fc_input] + fc_hidden_ls + [dim_model_output]

        self.n_linear_layers = len(fc_neuron_ls) - 1

        fc_block = []
        for layer_idx in range(self.n_linear_layers):
            dim_input = fc_neuron_ls[layer_idx]
            dim_output = fc_neuron_ls[layer_idx + 1]

            # normalization
            if fc_norm and (layer_idx != 0 or norm_fc_input):
                fc_block.append(nn.LayerNorm(dim_input))

            # linear connection
            fc_block.append(nn.Linear(dim_input, dim_output))

            # for non-output layers
            if layer_idx != (self.n_linear_layers-1):

                # dropout
                if fc_dropout_rate:
                    fc_block.append(nn.Dropout(p=fc_dropout_rate))

                # activation
                fc_block.append(nn.LeakyReLU())

        if norm_fc_output:
            fc_block.append(nn.LayerNorm(dim_output))

        self.fc_block = nn.Sequential(*fc_block)

    def forward(self, data_batch):
        '''Make connects between the components to complete the model'''

        ################################################################
        # GRAPH INPUT PREPARATION
        ################################################################

        # batch metadata
        batch_vector = data_batch['residue'].batch.long()

        # gather all inputs for GNN
        if self.graph_dims != []:
            node_feat = getattr(data_batch['residue'], self.node_feat_name).float()
            graph_input = [node_feat]

        # pipe node features to linear layers
        node_embeddings = []

        ################################################################
        # PI EMBEDDING
        ################################################################
        if self.use_pi:
            pi_embedding = self.pi_block(data_batch.pi.float())
        else:
            pi_embedding = None

        ################################################################
        # GRAPH CONVOLUTIONS
        ################################################################

        if self.graph_dims != []:
            graph_input = torch.cat(graph_input, dim=1)

        # pass each graph dimension through its own conv block
        for dim_idx, dim_name in enumerate(self.graph_dims):
            edge_type = ('residue', dim_name, 'residue')

            # if self.graph_dims == 'backbone':
            #     pass

            dim_edge_index = data_batch[edge_type].edge_index.long()

            # drop edges
            if self.dropedge_rate:
                dim_edge_index, _ = pyg.utils.dropout_edge(
                    dim_edge_index,
                    p=self.dropedge_rate,
                    force_undirected=True,
                    training=self.training
                )

            # drop nodes
            if self.dropnode_rate:
                raise NotImplementedError
                dim_edge_index, _, node_mask = pyg.utils.dropout_node(
                    dim_edge_index,
                    p=self.dropnode_rate,
                    num_nodes=data_batch['residue'].num_nodes,
                    training=self.training
                )

                # keep features only for retained nodes
                graph_input = graph_input[node_mask]

                # update batch vector to match new number of nodes
                batch_vector = batch_vector[node_mask]

            # pipe features from each graph dimension into the fc layer
            node_embeddings.append(
                self.conv_block_dict[dim_name](
                    graph_input,
                    dim_edge_index,
                    batch_vector
                )
            )

        if self.graph_dims != []:
            # concatenate node embeddings across dimensions
            node_embeddings = torch.cat(node_embeddings, dim=1)
            graph_embedding = self.graph_pool(
                node_embeddings, batch_vector
            )
        else:
            graph_embedding = None

        ################################################################
        # FC INPUT PREPARATION
        ################################################################

        # concatenate embeddings
        fc_input = torch.cat(
            [
                e for e in [
                    graph_embedding,
                    pi_embedding
                ]
                if e is not None
            ], dim=1
        )

        ################################################################
        # FC LAYERS
        ################################################################
        x = self.fc_block(fc_input)

        return x

    def save_args(self, save_dir):
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

    def reset_parameters(self):

        # (re)initialize convolutional parameters
        for dim_name in self.graph_dims:
            for layer in self.conv_block_dict[dim_name].children():
                if isinstance(layer, pyg.nn.conv.MessagePassing):
                    layer.reset_parameters()
                    # nn.init.kaiming_normal_(layer.lin.weight, a=0.01)
                    # nn.init.zeros_(layer.bias)
            # for name, param in mods.named_parameters():
            #     print(name, param.size())

        # (re)initialize fc parameters
        count = 1
        for layer in self.fc_block.children():
            if isinstance(layer, nn.Linear):
                if count < self.n_linear_layers:
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                    count += 1
                else:
                    nn.init.normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()

    def eval(self):
        return self.train(False)

    def train(self, mode=True):

        if mode:
            self.training = True
        else:
            self.training = False

        # call parent function
        return super().train(mode)

class mGCN(nn.Module):
    def __init__(
        self,
        dim_model_output,

        # FEATURE SELECTION
        graph_dims,
        use_pi=False,

        # GRAPH CONVOLUTION SETUP
        node_feat_name=None,
        node_feat_size=None,
        dim_node_hidden_ls=None,
        mgcn_hidden_channels=None,
        conv_norm=None,
        norm_graph_input=None,
        norm_graph_output=None,
        graph_global_pool=None,
        graph_dropout_rate=None,
        dropfeat_rate=None,
        dropedge_rate=None,
        dropnode_rate=None,
        jk_concat=None,

        # PERSISTENCE IMAGES SETUP
        pi_dropout_rate=None,
        dim_pi_embedding=None,

        # FC SETUP
        fc_hidden_ls=None,
        n_fc_hidden_layers=None,
        fc_norm=True,
        norm_fc_input=False,
        fc_dropout_rate=0.5,

        # OTHERS
        debug=False
        ):

        if debug:
            torch.autograd.set_detect_anomaly(True)

        ################################################################
        # SAVE A COPY OF ARGUMENTS PASSED
        ################################################################

        self.all_args = locals()
        del self.all_args['self'], self.all_args['__class__']

        # arguments required in forward()
        self.graph_dims = graph_dims
        self.use_pi = use_pi
        self.dropedge_rate = dropedge_rate
        self.dropnode_rate = dropnode_rate
        self.node_feat_name = node_feat_name

        # internal parameters
        self.training = True
        self.debug = debug

        super().__init__()

        ################################################################
        # INSTANTIATE POOLING LAYERS
        ################################################################
        if graph_global_pool is None:
            self.graph_pool = None
        elif graph_global_pool == 'mean':
            self.graph_pool = pyg.nn.global_mean_pool
        elif graph_global_pool == 'max':
            self.graph_pool = pyg.nn.global_max_pool
        elif graph_global_pool == 'sum':
            self.graph_pool = pyg.nn.global_add_pool
        else:
            raise ValueError(
                f'`graph_global_pool` must be "mean", "max", or "sum", '
                f'not {graph_global_pool}'
            )

        ################################################################
        # KEEP TRACK OF THE NUMBER OF FEATURE FOR FC LAYERS
        ################################################################

        dim_fc_input = 0

        ################################################################
        # INSTANTIATE GRAPH CONVOLUTIONAL LAYERS
        ################################################################

        n_dims = len(self.graph_dims)

        assert n_dims > 1

        mods = []

        dim_node_ls = [ node_feat_size ] + dim_node_hidden_ls
        dim_node_hidden = dim_node_ls[-1]

        total_node_dims = 0

        for layer_idx in range(len(dim_node_hidden_ls)):
            dim_input = dim_node_ls[layer_idx]
            dim_output = dim_node_ls[layer_idx + 1]

            next_idx = layer_idx

            # only for the first layer
            if layer_idx == 0:
                # dropfeat
                if dropfeat_rate:
                    mods.append((
                        nn.Dropout(p=dropfeat_rate),
                        f'x{next_idx} -> x{layer_idx+1}'
                    ))
                    next_idx = layer_idx+1

            # exclude the first layer
            if layer_idx != 0 or norm_graph_input:
                # normalization
                if conv_norm:
                    mods.append((
                        pyg.nn.GraphNorm(dim_input),
                        f'x{next_idx}, batch -> x{layer_idx+1}'
                    ))
                    next_idx = layer_idx+1

            # convolution
            conv = mGCNConv(
                in_channels=dim_input, out_channels=dim_output,
                hidden_channels=mgcn_hidden_channels, n_dims=n_dims
            )
            mods.append((
                conv,
                f'x{next_idx}, all_dim_edge_index -> x{layer_idx+1}'
            ))

            # dropout
            if graph_dropout_rate:
                mods.append((
                    nn.Dropout(p=graph_dropout_rate),
                    f'x{layer_idx+1} -> x{layer_idx+1}'
                ))

            # # activation
            # mods.append((
            #     nn.LeakyReLU(),
            #     f'x{layer_idx+1} -> x{layer_idx+1}'
            # ))

        feats = [f'x{i+1}' for i in range(len(dim_node_hidden_ls))]

        if jk_concat:
            # jumping knowledge connections
            mods.append((lambda *x: [*x], ', '.join(feats)+' -> xs'))

            mods.append((pyg.nn.JumpingKnowledge('cat'), 'xs -> x'))
            graph_embedding_size = sum(dim_node_hidden_ls)

            # normalization for concatenated node embeddings
            if norm_graph_output:
                mods.append(
                    (pyg.nn.GraphNorm(graph_embedding_size),
                        'x, batch -> x')
                )

            # sum size of embeddings accross all layers and graph dims
            total_node_dims += graph_embedding_size

        else:

            # normalization for node embeddings
            if norm_graph_output:
                mods.append(
                    (pyg.nn.GraphNorm(dim_node_hidden),
                        f'x{feats[-1]}, batch -> x{feats[-1]}')
                )

            # no jumping knowledge connections
            total_node_dims += dim_node_hidden

        # for m in mods:
        #     print(m)

        self.conv_block = (
            pyg.nn.Sequential('x0, all_dim_edge_index, batch', mods)
        )

        dim_fc_input += total_node_dims

        ################################################################
        # INSTANTIATE GRAPH CONVOLUTIONAL LAYERS
        ################################################################
        if use_pi:
            self.pi_block = nn.Sequential(
                nn.Linear(625, dim_pi_embedding),
                nn.LayerNorm(dim_pi_embedding),
                nn.Dropout(p=pi_dropout_rate),
                nn.LeakyReLU(),
            )
            dim_fc_input += dim_pi_embedding

        ################################################################
        # INSTANTIATE FULLY CONNECTED LAYERS
        ################################################################

        if fc_hidden_ls is None:
            factor = dim_fc_input//(n_fc_hidden_layers+1)
            if factor !=0:
                fc_hidden_ls = [
                    factor*i for i in range(1,n_fc_hidden_layers+1)[::-1]
                ]
            else:
                fc_hidden_ls = [1] * n_fc_hidden_layers

        fc_neuron_ls = [dim_fc_input] + fc_hidden_ls + [dim_model_output]

        self.n_linear_layers = len(fc_neuron_ls) - 1

        fc_block = []
        for layer_idx in range(self.n_linear_layers):
            dim_input = fc_neuron_ls[layer_idx]
            dim_output = fc_neuron_ls[layer_idx + 1]

            # normalization
            if fc_norm and (layer_idx != 0 or norm_fc_input):
                fc_block.append(nn.BatchNorm1d(dim_input, affine=True))

            # linear connection
            fc_block.append(nn.Linear(dim_input, dim_output))

            # for non-output layers
            if layer_idx != (self.n_linear_layers-1):

                # dropout
                if fc_dropout_rate:
                    fc_block.append(nn.Dropout(p=fc_dropout_rate))

                # activation
                fc_block.append(nn.LeakyReLU())

        self.fc_block = nn.Sequential(*fc_block)

    def forward(self, data_batch):
        '''Make connects between the components to complete the model'''

        ################################################################
        # GRAPH INPUT PREPARATION
        ################################################################

        # batch metadata
        batch_vector = data_batch['residue'].batch.long()

        # gather all inputs for GNN
        if self.graph_dims != []:
            node_feat = getattr(data_batch['residue'], self.node_feat_name).float()
            graph_input = [node_feat]

        # # pipe node features to linear layers
        # node_embeddings = []

        ################################################################
        # PI EMBEDDING
        ################################################################
        if self.use_pi:
            pi_embedding = self.pi_block(data_batch.pi.float())
        else:
            pi_embedding = None

        ################################################################
        # GRAPH CONVOLUTIONS
        ################################################################

        if self.graph_dims != []:
            graph_input = torch.cat(graph_input, dim=1)

        # pass each graph dimension through its own conv block
        all_dim_edge_index = [
            data_batch[('residue', dim_name, 'residue')].edge_index.long()
            for dim_name in self.graph_dims
        ]

        # # drop edges
        # if self.dropedge_rate:
        #     dim_edge_index, _ = pyg.utils.dropout_edge(
        #         dim_edge_index,
        #         p=self.dropedge_rate,
        #         force_undirected=True,
        #         training=self.training
        #     )

        # drop nodes
        if self.dropnode_rate:
            dim_edge_index, _, node_mask = pyg.utils.dropout_node(
                dim_edge_index,
                p=self.dropnode_rate,
                num_nodes=data_batch['residue'].num_nodes,
                training=self.training
            )

            # keep features only for retained nodes
            graph_input = graph_input * node_mask[:, None]

            # update batch vector to match new number of nodes
            batch_vector = batch_vector[node_mask]

        # pipe features from each graph dimension into the fc layer
        node_embeddings = self.conv_block(
            graph_input,
            all_dim_edge_index,
            batch_vector
        )

        if self.graph_dims != []:
            # concatenate node embeddings across dimensions
            # node_embeddings = torch.cat(node_embeddings, dim=1)
            graph_embedding = self.graph_pool(
                node_embeddings, batch_vector
            )
        else:
            graph_embedding = None

        ################################################################
        # FC INPUT PREPARATION
        ################################################################

        # concatenate embeddings
        fc_input = torch.cat(
            [
                e for e in [
                    graph_embedding,
                    pi_embedding
                ]
                if e is not None
            ], dim=1
        )

        ################################################################
        # FC LAYERS
        ################################################################
        x = self.fc_block(fc_input)

        return x

    def save_args(self, save_dir):
        with open(path.join(save_dir, 'model-args.json'),
                  'w+') as f_out:
            json.dump(self.all_args, f_out,
                      indent=4, separators=(',', ': '), sort_keys=True)

        with open(path.join(save_dir, 'model-summary.txt'),
                  'w+', encoding='utf-8') as sys.stdout:
            print(self, end='\n\n')
            summary(self)
        sys.stdout = sys.__stdout__

    def reset_parameters(self):

        # (re)initialize convolutional parameters
        for dim_name in self.graph_dims:
            for layer in self.conv_block.children():
                if isinstance(layer, pyg.nn.conv.MessagePassing):
                    layer.reset_parameters()
                    # nn.init.kaiming_normal_(layer.lin.weight, a=0.01)
                    # nn.init.zeros_(layer.bias)
            # for name, param in mods.named_parameters():
            #     print(name, param.size())

        # (re)initialize fc parameters
        count = 1
        for layer in self.fc_block.children():
            if isinstance(layer, nn.Linear):
                if count < self.n_linear_layers:
                    nn.init.kaiming_normal_(
                        layer.weight, a=0.01, nonlinearity='leaky_relu'
                    )
                    nn.init.zeros_(layer.bias)
                    count += 1
                else:
                    nn.init.normal_(layer.weight)
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                layer.reset_parameters()

    def eval(self):
        return self.train(False)

    def train(self, mode=True):

        if mode:
            self.training = True
        else:
            self.training = False

        # call parent function
        return super().train(mode)

if __name__ == '__main__':

    import torchinfo

    model = mGCN(
        # FEATURE SELECTION
        graph_dims=['cont', 'deform'],
        use_pi=False,

        # GRAPH CONVOLUTION SETUP
        node_feat_name='x',
        node_feat_size=1024,
        dim_node_hidden_ls=[64, 64],
        conv_norm=True,
        norm_graph_input=False,
        norm_graph_output=False,
        graph_global_pool='mean',
        graph_dropout_rate=0,
        dropfeat_rate=0,
        dropedge_rate=0,
        dropnode_rate=0,
        jk_concat=None,

        # PERSISTENCE IMAGES SETUP
        pi_dropout_rate=0,
        dim_pi_embedding=32,

        # FC SETUP
        fc_hidden_ls=None,
        n_fc_hidden_layers=2,
        fc_norm=True,
        norm_fc_input=False,
        fc_dropout_rate=0,

        # OTHERS
        debug=False
    )

    # model = SimplifiedMultiGCN(
    #     # FEATURE SELECTION
    #     # graph_dims=['cont'],
    #     use_pi=True,

    #     # GRAPH CONVOLUTION SETUP
    #     node_feat_name='x',
    #     node_feat_size=1024,
    #     gnn_type='gcn',
    #     gat_atten_heads=None,
    #     dim_node_hidden_dict={
    #         'contact': [32,32],
    #         # 'codir': [32],
    #         # 'coord': [32],
    #         'deform': [32]
    #     },
    #     conv_norm=True,
    #     norm_graph_input=False,
    #     norm_graph_output=False,
    #     graph_global_pool='mean',
    #     graph_dropout_rate=0,
    #     dropfeat_rate=0,
    #     dropedge_rate=0,
    #     dropnode_rate=0,
    #     jk_concat=None,

    #     # PERSISTENCE IMAGES SETUP
    #     pi_dropout_rate=0.2,
    #     dim_pi_embedding=32,

    #     # FC SETUP
    #     fc_hidden_ls=None,
    #     n_fc_hidden_layers=2,
    #     fc_norm=True,
    #     norm_fc_input=False,
    #     fc_dropout_rate=0.5,

    #     # OTHERS
    #     debug=False
    # )

    print()
    print(model)
    print()
    torchinfo.summary(model)
