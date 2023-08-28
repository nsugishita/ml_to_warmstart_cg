# -*- coding: utf-8 -*-

"""Neural network initialiser"""

try:
    import torch

    torch.set_num_threads(1)

    has_torch = True
    torch_nn_Module = torch.nn.Module

except ImportError:
    has_torch = False
    torch_nn_Module = object

from sucpy.dual_initialisers.base import DualInitialiser
from sucpy.utils import network_utils


class BaseNetworkInitialiser(DualInitialiser, torch_nn_Module):  # type: ignore
    """Neural network initialiser"""

    def __init__(
        self,
        dim_parameter,
        dim_dual,
        config={},
        network_kwargs={},
    ):
        """Initialise a BaseNetworkInitialiser instance

        Parameters
        ----------
        dim_parameter : int
        dim_dual : int
        config : dict, optional
        """
        if not has_torch:
            raise ImportError("torch")
        super().__init__()
        self.config = config
        self.dim_parameter = dim_parameter
        self.dim_dual = dim_dual
        self.network = network_utils.NeuralNetwork(
            self.dim_parameter, self.dim_dual, **network_kwargs
        )
        self.parameter_normalizer = network_utils.Normalizer(
            self.dim_parameter
        )
        self.dual_normalizer = network_utils.Normalizer(self.dim_dual)

    def init_weight(self):
        """Initialise the weights"""
        self.network.init_weight()

    def set_normalizer(self, parameter, dual):
        """Set internal normalizers based on given data

        This sets internal normalizers.
        This should be called before any training.
        Once this is called, the state of normalizer is saved in
        state_dict, so one can load and use network immediately.
        """
        parameter = torch.Tensor(parameter)
        dual = torch.Tensor(dual)
        self.parameter_normalizer.set(parameter)
        self.dual_normalizer.set(dual)

    def load(self, path):
        """Load parameters from a disk"""
        if not path.endswith("pth"):
            raise ValueError
        self.load_state_dict(torch.load(path))

    def share_memory(self):
        """Share memory of the underlying network"""
        self.network.share_memory()

    def forward(self, parameter):
        """Output dual based on current policy for training purpose

        Parameters
        ----------
        parameter : (dim_parameter,) or (batch_size, dim_parameter) tensor

        Returns
        -------
        dual : (dim_dual,) or (batch_size, dim_dual) tensor
            Predicted dual if not a3c or mean of dual if a3c.
        """
        parameter = torch.Tensor(parameter)
        if parameter.ndimension() == 1:
            parameter = parameter[None]
            batch = False
        else:
            batch = True
        parameter = self.parameter_normalizer.normalize(parameter)
        dual = self.network(parameter)
        if not batch:
            dual = dual.squeeze(dim=0)
        dual = self.dual_normalizer.denormalize(dual)
        return dual

    def compute_initial_dual(self, parameter):
        """Output dual for CG

        This is used within CG.  Note that this returns
        a numpy array instead of a tensor.

        Parameters
        ----------
        parameter : (dim_parameter,) array

        Returns
        -------
        dual : (dim_dual,) array
        """
        with torch.no_grad():
            return self(parameter).numpy()


class SingleSamplingNetworkInitialiser(BaseNetworkInitialiser):
    pass


class DoubleSamplingNetworkInitialiser(BaseNetworkInitialiser):
    pass
