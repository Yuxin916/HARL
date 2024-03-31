import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from harl.utils.models_tools import init, get_active_func, get_init_method


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, initialization_method, activation_func):
        """Initialize the GCN layer.

        Args:
            input_dim: (int) Input dimension.
            output_dim: (int) Output dimension.
            initialization_method: (str) Initialization method.
            activation_func: (str) Activation function.
        """
        super(GCNLayer, self).__init__()

        active_func = get_active_func(activation_func)  # Activation function
        init_method = get_init_method(initialization_method)  # Initialization method

        # Adjusting the weight scale to ensure that activations in the layer do not become too small (vanishing gradient) or too large (exploding gradient) during training
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        # Create the GCN layer
        self.gcn = init_(GCNConv(input_dim, output_dim))
        self.active_func = active_func

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        x = self.active_func(x)
        return x


class GCNBase(nn.Module):
    """A GCN base module."""

    def __init__(self, args, input_dim):
        super(GCNBase, self).__init__()

        self.initialization_method = args["initialization_method"]  # Initialization method
        self.activation_func = args["activation_func"]  # Activation function
        self.output_dim = args["output_dim"]  # Output dimension for GCN layer

        # Create GCN layer with specified input_dim, output_dim, initialization method, and activation function
        self.gcn = GCNLayer(
            input_dim, self.output_dim, self.initialization_method, self.activation_func
        )

    def forward(self, x, edge_index):
        x = self.gcn(x, edge_index)
        return x
