import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torchinfo import summary
# from torch_geometric.utils import add_self_loops
# from torch_geometric.nn.conv import MessagePassing

# class mGCNConv(MessagePassing):
class mGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, n_dims):
        super().__init__()

        self.n_dims = n_dims

        self.proj_mat = nn.parameter.Parameter(
            torch.zeros(n_dims, in_channels, hidden_channels)
        )
        self.act1 = nn.LeakyReLU()

        self.within_gcn = nn.ModuleList([])
        for _ in range(n_dims):
            self.within_gcn.append(
                GCNConv(hidden_channels, hidden_channels)
            )

        self.bilinear = nn.parameter.Parameter(
            torch.zeros(hidden_channels, hidden_channels)
        )
        # importance wrt dim i is in column i
        self.softmax = nn.Softmax(dim=0)

        # self.alpha = nn.Parameter(
        #     torch.zeros(n_dims)
        # )
        self.alpha = 0.5

        self.inv_proj = nn.Linear(
            hidden_channels * n_dims, out_channels, bias=True
        )
        self.act2 = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self):
        # super().reset_parameters()
        nn.init.kaiming_uniform_(self.proj_mat)
        [self.within_gcn[i].reset_parameters() for i in range(self.n_dims)]
        nn.init.kaiming_uniform_(self.bilinear)
        if isinstance(self.alpha, nn.Parameter):
            nn.init.constant_(self.alpha, 0.5)
        self.inv_proj.reset_parameters()

    def forward(self, x, all_dim_edge_index):

        '''
        INPUTS

            x (tensor, shape: [N, in_channels])
                general representation of the nodes

            all_dim_edge_index (list of n_dims tensors with shape: [2, E])
                edge connections of each graph dimension
        '''

        # # check that all_dim_edge_index has the correct number of dimensions
        # if not len(all_dim_edge_index) == self.n_dims:
        #     raise RuntimeError(
        #         f'first dim of all_dim_edge_index should have size {self.n_dims}, '
        #         f'but got {all_dim_edge_index.shape} instead'
        #     )

        # # Step 1: Add self-loops to the adjacency matrix.
        # for i in range(self.n_dims):
        #     all_dim_edge_index[i], _ = add_self_loops(all_dim_edge_index[i],
        #                                           num_nodes=x.size(0))

        # Step 2: Project general representation to individual dims.
        # shape: (n_dims, N, hidden_channels)
        dim_repr = self.act1(torch.matmul(x, self.proj_mat))

        # Step 3-1: within-dimension aggregation (GCN)
        within_aggr = [] # n_dims items with shape (N, hidden_channels)
        for i in range(self.n_dims):
            within_aggr.append(
                self.within_gcn[i](dim_repr[i], all_dim_edge_index[i])
            )
        # for i in range(self.n_dims):
        #     print(within_aggr[i].shape)
        # shape: (n_dims, N, hidden_channels)
        within_aggr = torch.stack(within_aggr)

        # Step 3-2: across-dimension aggregation
        attention = torch.einsum('aji,ik,bjk->ab',
                                 self.proj_mat, self.bilinear, self.proj_mat)
        # shape: (n_dims, n_dims)
        attention = self.softmax(attention)
        # shape: (n_dims, N, hidden_channels)
        across_aggr = torch.tensordot(attention, dim_repr, dims=([0],[0]))

        # Step 4: Sum within- and acros-aggr for dim-specific repr
        # shape: (n_dims, N, hidden_channels)
        if isinstance(self.alpha, nn.Parameter):
            new_dim_repr = (
                torch.tensordot(self.alpha, within_aggr, dims=([0],[0])) +
                torch.tensordot((1-self.alpha), across_aggr, dims=([0],[0]))
            )
        else:
            new_dim_repr = self.alpha * within_aggr + (1-self.alpha) * across_aggr

        # Step 5: Combine all dim-specific repr for new general repr
        # shape: (N, n_dims * hidden_channels)
        # concat = torch.dstack(torch.split(new_dim_repr, 1)).squeeze()
        concat = new_dim_repr.permute(1,0,2).reshape(new_dim_repr.shape[1], -1)
        # shape: (N, out_channels)
        x = self.act2(self.inv_proj(concat))
        # x = self.inv_proj(concat)

        return x

# class Test(nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.proj_mat = nn.parameter.Parameter(
#             torch.zeros(3, 5, 7)
#         )

#     def forward(self, x):
#         return torch.matmul(x, self.proj_mat)

if __name__ == '__main__':

    mgcn = mGCNConv(
        in_channels=21,
        hidden_channels=128,
        out_channels=128,
        n_dims=3
    )
    # mgcn = Test()
    print(mgcn)
    summary(mgcn)
