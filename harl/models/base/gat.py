import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from harl.utils.models_tools import init, get_active_func, get_init_method

class GATLayer(nn.Module):
    def __init__(self, input_dim, output_dim, heads, initialization_method, activation_func):
        """Initialize the GAT layer.

        Args:
            input_dim: (int) Input dimension.
            output_dim: (int) Output dimension.
            heads: (int) Number of attention heads.
            initialization_method: (str) Initialization method.
            activation_func: (str) Activation function.
        """
        super(GATLayer, self).__init__()

        active_func = get_active_func(activation_func)  # Activation function
        init_method = get_init_method(initialization_method)  # Initialization method

        # Adjusting the weight scale
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # Create the GAT layer
        self.gat = init_(GATConv(input_dim, output_dim, heads=heads))
        self.active_func = active_func

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        x = self.active_func(x)
        return x


class GATBase(nn.Module):
    """A GAT base module."""

    def __init__(self, args, input_dim):
        super(GATBase, self).__init__()

        self.initialization_method = args["initialization_method"]  # Initialization method
        self.activation_func = args["activation_func"]  # Activation function
        self.output_dim = args["output_dim"]  # Output dimension for GAT layer
        self.heads = args["heads"]  # Number of attention heads

        # Create GAT layer with specified input_dim, output_dim, heads, initialization method, and activation function
        self.gat = GATLayer(
            input_dim, self.output_dim, self.heads, self.initialization_method, self.activation_func
        )

    def forward(self, x, edge_index):
        x = self.gat(x, edge_index)
        return x
