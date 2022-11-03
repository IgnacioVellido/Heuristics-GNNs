from torch import nn, min #, cat, relu
from torch_geometric.nn.conv import MessagePassing

class RACGNN(nn.Module):
    """Modified ACGNN"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            aggregate_type: str,
            combine_type: str,
            combine_layers: int,
            num_mlp_layers: int,
            final_mlp: bool,
            **kwargs
    ):
        super(ACGNN, self).__init__()

        self.final_mlp = final_mlp

        # self.bigger_input = input_dim > hidden_dim
        self.mlp_combine = combine_type == "mlp"

        self.activation = nn.ReLU()

        # MOD: Only one layer
        self.conv = ACConv(input_dim=input_dim,
                           output_dim=hidden_dim,
                           aggregate_type=aggregate_type,
                           combine_type=combine_type,
                           combine_layers=combine_layers,
                           num_mlp_layers=num_mlp_layers)

        if self.final_mlp:
            # 1 layer
            # self.linear_prediction = nn.Linear(hidden_dim, output_dim)

            # >1 layer
            self.linear_prediction = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )


    def forward(self, x, edge_index, batch, passes=1):
        h = x
        # if not self.bigger_input:
        #     h = self.padding(h)

        # MOD: Indicate passes as argument
        for p in range(passes):
            # MOD: calling the same convs
            h = self.conv(h=h, edge_index=edge_index, batch=batch)

            # This gives poorer results
            h = self.activation(h)
            # h = self.batch_norm(h)

        if self.final_mlp:
            return self.linear_prediction(h)
        else:
            return h


    def reset_parameters(self):
        reset(self.convs)
        reset(self.batch_norms)
        reset(self.linear_prediction)

# -------------------------------------------------------------

class ACConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            combine_type: str,
            combine_layers: int,
            num_mlp_layers: int,
            **kwargs):

        # MOD: added "min"
        assert aggregate_type in ["add", "mean", "max", "min"]
        assert combine_type in ["simple", "mlp"]

        super(ACConv, self).__init__(aggr=aggregate_type, **kwargs)

        self.mlp_combine = False
        if combine_type == "mlp":
            self.mlp = MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)

            self.mlp_combine = True

        self.V = MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)
        self.A = MLP(
            num_layers=combine_layers,
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim)


    def forward(self, h, edge_index, batch):
        return self.propagate(
            edge_index=edge_index,
            h=h)


    def message(self, h_j):
        return h_j
        # return h_j + edge_weight

    # With edge_weights
    # def message(self, h_j: Tensor, edge_weight: OptTensor) -> Tensor:
    #     return h_j if edge_weight is None else (edge_weight.view(-1, 1) * h_j)


    def update(self, aggr, h):
        # MOD: concat, not sum
        # updated = self.V(h) + self.A(aggr)
        # updated = cat([self.V(h), self.A(aggr)], dim=1)

        updated = self.V(h) + min(h, self.A(aggr))
        # updated = self.V(h) - self.V(-h) - self.A(h-aggr)

        if self.mlp_combine:
            updated = self.mlp(updated)

        # Residuals
        # updated = h + updated

        return updated

    def reset_parameters(self):
        self.V.reset_parameters()
        self.A.reset_parameters()
        if hasattr(self, "mlp"):
            self.mlp.reset_parameters()

# -------------------------------------------------------------

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)



class MLP(nn.Module):

    # MLP with linear output
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        self.activation = nn.ReLU()

        if num_layers < 1:
            self.linear = nn.Identity()
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = nn.ModuleList()
            self.batch_norms = nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            # for layer in range(num_layers - 1):
            #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            
            # self.batch_norms.append(nn.BatchNorm1d((output_dim)))

            # for layer in self.linears:
            #     nn.init.xavier_uniform(layer.weight)


    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            x = self.linear(x)
            x = self.activation(x) # Mod
            return x
        else:
            # If MLP
            h = x
            
            for layer in range(self.num_layers - 1):
                # h = relu(self.batch_norms[layer](self.linears[layer](h)))
                h = self.activation(self.linears[layer](h))

            return self.linears[self.num_layers - 1](h)


    def reset_parameters(self):
        if self.linear_or_not:
            reset(self.linear)
        else:
            reset(self.linears)
            reset(self.batch_norms)