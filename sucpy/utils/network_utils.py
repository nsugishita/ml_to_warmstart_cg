# -*- coding: utf-8 -*-

"""Utils to define neural networks."""

import collections
import numbers

import numpy as np

try:
    import torch

    torch.set_num_threads(1)

    has_torch = True
    torch_nn_Module = torch.nn.Module

except ImportError:
    has_torch = False
    torch_nn_Module = object


class NeuralNetwork(torch_nn_Module):  # type: ignore
    """Neural network with skip connections between hidden layers"""

    def __init__(
        self,
        dim_input,
        dim_output,
        layer_sizes=[1000, 1000, 1000, 1000],
        last_layer_weight=0.003,
        skip_init=0.0,
        activation="tanh",
        input_dropout=None,
        hidden_layer_dropout=None,
    ):
        """Initialise a NeuralNetwork instance

        Parameters
        ----------
        dim_input : int
        dim_output : int or tuple of int
        layer_sizes : list of ints, default [200, 200]
        last_layer_weight : float, default 0.003
            Initialise last layer weight by uniform distribution
            U[-last_layer_weight, last_layer_weight] if given.
        skip_init : float or None, default 0.0
            Initial value of the scalars multiplied to
            the output of the residual branches.
            Setting this to None suppress the use of
            skipinit.
        input_dropout : float, optional
            The rate of dropout on input.
        hidden_layer_dropout : float, optional
            The rate of dropout on hidden layers.
        """
        if not has_torch:
            raise ImportError("torch")
        super().__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_hidden_layers = len(layer_sizes)
        self.activation = get_activation(activation)
        self.last_layer_weight = last_layer_weight

        # Set attributes.
        sz = [int(x) for x in layer_sizes]
        sz.insert(0, int(dim_input))
        sz.append(int(np.prod(dim_output)))

        # Define all layers in torch.nn.Modulelist.
        # If we hold the layers in a python native list, instead of ModuleList,
        # then the parameters are not handled correctly.
        layers = [
            torch.nn.Linear(sz[i], sz[i + 1]) for i in range(len(sz) - 1)
        ]
        self.layers = torch.nn.ModuleList(layers)

        # Factors multiplied to the output of the residual branches.
        Parameter = torch.nn.parameter.Parameter
        if skip_init is not None:
            self.residual_branches_output_scalers = Parameter(
                torch.full((len(layers) - 2,), skip_init),
                requires_grad=True,
            )
        else:
            self.residual_branches_output_scalers = Parameter(
                torch.full((len(layers) - 2,), 1.0),
                requires_grad=False,
            )

        if input_dropout is not None and input_dropout > 0:
            self.input_dropout = torch.nn.Dropout(p=input_dropout)
        else:
            self.input_dropout = None

        if hidden_layer_dropout is not None and hidden_layer_dropout > 0:
            self.hidden_layer_dropout = torch.nn.Dropout(
                p=hidden_layer_dropout
            )
        else:
            self.hidden_layer_dropout = None

        self.init_weight()

    def init_weight(self):
        """Initialise the weights"""
        # Initialise whole parameters (weights).
        self.apply(init_weight)
        # If last layer weight is given explicitly, use that value.
        w = self.last_layer_weight
        if (w is not None) and (w > 0):
            self.layers[-1].weight.data.uniform_(-w, w)
            self.layers[-1].bias.data.fill_(0)

    def forward(self, inputs):
        """Compute the forward path

        Parameters
        ----------
        inputs : (num_batches, dim_input) tensor

        Returns
        -------
        outputs : (num_batches, dim_output) tensor
        """
        xx = inputs
        for ii in range(len(self.layers)):
            # Apply dropout.
            if ii == 0 and self.input_dropout:
                xx = self.input_dropout(xx)
            elif ii >= 1 and self.hidden_layer_dropout:
                xx = self.hidden_layer_dropout(xx)
            skipped = xx
            # Compute the value output from this layer.
            xx = self.layers[ii](xx)
            if ii != len(self.layers) - 1:
                xx = self.activation(xx)
            if 0 < ii < len(self.layers) - 1:
                alpha = self.residual_branches_output_scalers[ii - 1]
                xx = alpha * xx + skipped  # Add the skip connection.
        if isinstance(self.dim_output, (tuple, list)):
            batch_shape = xx.shape[:-1]
            out_shape = batch_shape + tuple(self.dim_output)
            xx = xx.reshape(out_shape)
        return xx


def init_weight(m, gain=1.0):
    """Initialise the linear layers' weight with Xavier uniform"""
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, torch.nn.ModuleList):
        for w in m:
            init_weight(w, gain=gain)


_missing = object()


def yield_with_next(seq, last=_missing):
    """Yield values from a given sequence by pairs

    Note that the last element is not yielded when one does
    not specify `last`.

    Examples
    --------
    >>> for l, r in yield_with_next(range(4)):
    ...     print(l, ':', r)
    0 : 1
    1 : 2
    2 : 3

    >>> for l, r in yield_with_next(range(4), last=-1):
    ...     print(l, ':', r)
    0 : 1
    1 : 2
    2 : 3
    3 : -1

    Parameters
    ----------
    seq : iterable
    last : obj, optional
        object returns with the last element.
        If this is omitted, last element is not yielded.

    Yields
    ------
    pair : tuple of 2 objs
    """
    empty = {}
    previous = empty
    for ss in seq:
        if previous is not empty:
            yield (previous, ss)
        previous = ss
    if last is not _missing:
        yield (ss, last)


class Normalizer(torch_nn_Module):  # type: ignore
    """Base class for normalizer layers"""

    def __init__(self, *dim, mean=None, std=None):
        """Initialise an Normalizer instance"""
        if not has_torch:
            raise ImportError("torch")
        super().__init__()
        if len(dim) == 1 and isinstance(dim[0], tuple):
            dim = dim[0]
        self.dim = dim
        if mean is None:
            if dim:
                self.register_buffer("mean", torch.zeros(*dim))
            else:
                self.register_buffer("mean", torch.zeros(1)[0])
        else:
            self.register_buffer("mean", mean)
        if std is None:
            if dim:
                self.register_buffer("std", torch.ones(*dim))
            else:
                self.register_buffer("std", torch.ones(1)[0])
        else:
            self.register_buffer("std", std)

    def set(self, data, std_clip=1e-1):
        """Compute mean and std deviation on given data

        Parameters
        ----------
        data : (batch_size, *dim) tensor or array
        std_clip : float, default 1e-1
            Minimum threshold of std.  Std smaller than this value is
            clipped to this.
        """
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        self.mean = data.mean(dim=0)
        if std_clip is not None:
            self.std = data.std(dim=0).clamp(std_clip)
        else:
            self.std = data.std(dim=0)

    def normalize(self, input):
        """Normalize a given value"""
        return self.normalise(input)

    def normalise(self, input):
        """Normalise a given value"""
        mean = self.mean  # typically (dim_input,)
        std = self.std  # typically (dim_input,)
        if mean is not None and not torch.all(mean == 0):
            input = input - mean
        if std is not None and not torch.all(std == 1):
            input = input / std
        return input

    def denormalize(self, input):
        """De-normalize a given value"""
        return self.denormalise(input)

    def denormalise(self, input):
        """De-normalise a given value"""
        mean = self.mean  # typically (dim_input,)
        std = self.std  # typically (dim_input,)
        if std is not None and not torch.all(std == 1):
            input = input * std
        if mean is not None and not torch.all(mean == 0):
            input = input + mean
        return input


Normaliser = Normalizer


class TanhNormalizer(torch_nn_Module):  # type: ignore
    """Normalizer tanh squashing values within a prescribed range"""

    def __init__(
        self, *dim, min=None, max=None, mean=None, std=None, epsilon=1e-4
    ):
        """Initialise an TanhNormalizer instance

        Parameters
        ----------
        *dim : dimension of input
        min : (dim_input,) tensor
        max : (dim_input,) tensor
        """
        if not has_torch:
            raise ImportError("torch")
        super().__init__()
        if len(dim) == 1 and isinstance(dim[0], tuple):
            dim = dim[0]
        self.dim = dim
        if min is None:
            if dim:
                self.register_buffer("min", torch.zeros(*dim))
            else:
                self.register_buffer("min", torch.zeros(1)[0])
        else:
            self.register_buffer("min", min)
        if max is None:
            if dim:
                self.register_buffer("max", torch.zeros(*dim))
            else:
                self.register_buffer("max", torch.zeros(1)[0])
        else:
            self.register_buffer("max", max)
        if mean is None:
            if dim:
                self.register_buffer("mean", torch.zeros(*dim))
            else:
                self.register_buffer("mean", torch.zeros(1)[0])
        else:
            self.register_buffer("mean", mean)
        if std is None:
            if dim:
                self.register_buffer("std", torch.ones(*dim))
            else:
                self.register_buffer("std", torch.ones(1)[0])
        else:
            self.register_buffer("std", std)
        self.register_buffer("epsilon", torch.ones(1)[0])

    def share_memory(self):
        self.min.share_memory_()
        self.max.share_memory_()
        self.mean.share_memory_()
        self.std.share_memory_()

    def set(self, data, std_clip=1e-1):
        """Compute mean and std deviation on given data

        Parameters
        ----------
        data : (batch_size, *dim) tensor or array
        """
        if not isinstance(data, torch.Tensor):
            data = torch.Tensor(data)
        self.min.copy_(data.min(dim=0).values)
        self.max.copy_(data.max(dim=0).values)
        self.mean.copy_(data.mean(dim=0))
        std = data.std(dim=0)
        if std_clip is not None and std_clip > 0:
            std = std.clamp(std_clip)
        self.std.copy_(std)

    def normalize(self, input):
        """Normalize a given value"""
        return self.normalise(input)

    def normalise(self, input):
        """Normalise a given value"""
        # clamp input to lie between min and max
        input = input.max(self.min + self.epsilon).min(self.max - self.epsilon)
        # scale input to lie between 0 and 1
        input = (input - self.min) / (self.max - self.min)
        # scale input to lie between -1 and 1
        input = (input - 0.5) * 2
        return self._arctanh(input)

    def denormalize(self, input):
        """De-normalize a given value"""
        return self.denormalise(input)

    def denormalise(self, input):
        """De-normalise a given value"""
        # input lies between -1 and 1
        input = torch.tanh(input)
        # input lies between 0 and 1
        input = 0.5 * (input + 1)
        # input lies between min and max
        input = input * (self.max - self.min) + self.min
        return input

    def _arctanh(self, x):
        if isinstance(x, torch.Tensor):
            return 0.5 * torch.log((1 + x) / (1 - x))
        else:
            import numpy as np

            return 0.5 * np.log((1 + x) / (1 - x))

    def __str__(self):
        return f"{self.__class__.__name__}(range={self.mean})"


TanhNormaliser = TanhNormalizer


def get_activation(activation, *args, **kwargs):
    """Return pytorch activation instance from its name

    This returns appropriate pytorch activation from its name.
    If an object which is not str is given (including None),
    this returns it as it is.  Note name is case-insensitive.

    Parameters
    ----------
    activation : str or obj
    *args, **kwargs
        Passed to initialiser of an activation.
    """
    if not isinstance(activation, str):
        return activation
    elif activation.lower() == "none":
        return None
    else:
        activation = activation.lower()
        names = {n.lower(): getattr(torch.nn, n) for n in dir(torch.nn)}
        return names[activation](*args, **kwargs)


def zero_grad(module, set_to_none=False):
    """Sets the gradients of all optimized Tensors to zero

    This is taken from pytorch v1.7 (see Optimizer class therein).
    """
    for p in module.parameters():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                    p.grad.zero_()


def copy_grad(local_model, global_model):
    """Write gradients of a global model using local one

    This accepts a local model and a global model, and copy
    gradients computed on the local model to the global model.

    After this, one may call the optimizer on the global model
    and update the global parameters.

    This methods has to be called after the local model computed
    gradients.

    Parameters
    ----------
    local_model : torch.nn.Module
        Local model with 'thread-specific parameters'.
    global_model : torch.nn.Module
        Global model whose weights are shared among processes.
    """
    for param, shared_param in zip(
        local_model.parameters(), global_model.parameters()
    ):
        shared_param.grad = param.grad


def set_lr(optimizer, lr):
    """Update lr of optimizer"""
    for param in optimizer.param_groups:
        param["lr"] = lr


def get_lr(optimizer):
    """Update lr of optimizer"""
    for param in optimizer.param_groups:
        return param["lr"]


def multiply_lr(optimizer, factor):
    """Update lr of optimizer"""
    for param in optimizer.param_groups:
        param["lr"] *= factor


def get_batches(
    batch_size,
    sample=None,
    dropout=None,
    *arrays,
    index=False,
    yield_all=False,
    rng=None,
    **named_arrays,
):
    """Yield mini-batches

    Parameters
    ----------
    batch_size : int or float
        Size of each mini-batch.
        If float is given, it should be between 0 and 1 and
        batch_size is computed as the total number of data points
        times this value.
    sample : int or float, optional
        Size of samples to be used.  This determine how many samples
        are actually used.  If this value is smaller than the number
        of whole data, first this down samples data.
        If float is given, it should be between 0 and 1 and
        sample_size is computed as the total number of data points
        times this value.
        sample and dropout cannot be specified at the same time.
    dropout : float, optional
        If given, this must be between 0 and 1.
        This specifies an rate to drop the data;
        0.0 does not drop any samples at all and larger value
        drop samples more aggressively.
        sample and dropout cannot be specified at the same time.
    index : bool, default False
        Return corresponding index if True.
    yield_all : bool, default False
        If False (default), all mini-batches are exactly same size.
        In this case, if the total number of data is not divided by
        the batch size, some data are not yielded.
        If True, the last mini-batch may be smaller than the others
        but it is guaranteed that all data appears exactly once.
    rng : int, numpy.random.RandomState, optional
    *arrays : (num_samples, *) tensor or array
    **named_arrays : (num_samples, *) tensor or array

    Yields
    ------
    indices : tensor or array, optional
        This is only returned when `index` is True.
        Indices of data sampled.
    samples : (named) tuple of (batch_size, *) tensor or array
        Sampled arrays/tensors.  If keyword arguments are given,
        a named tuple is yielded.

    Examples
    --------
    >>> import numpy as np

    >>> x = np.random.rand(100, 10)
    >>> y = np.random.rand(100, 1, 20)
    >>> b = list(get_batches(30, x=x, y=y))
    >>> len(b)  # list of mini-batches.
    3
    >>> b0 = b[0]  # extract the first batch.
    >>> len(b0)
    2

    >>> b[0].x.shape  # mini-batch for x.
    (30, 10)
    >>> b[0].y.shape  # mini-batch for y.
    (30, 1, 20)

    >>> b = list(get_batches(0.2, x=x, y=y, dropout=0.5))  # Only 50 samples.
    >>> len(b)
    5
    >>> b0 = b[0]
    >>> len(b0.x)
    10
    """
    counter = -1

    given_arrays = collections.OrderedDict()

    for array in arrays:
        while True:
            counter += 1
            name = f"a{counter}"
            if name not in named_arrays:
                given_arrays[name] = array
                break

    given_arrays.update(named_arrays)

    if len(given_arrays) == 0:
        yield from ()  # If there is no data, exit early.
        return

    if isinstance(rng, numbers.Number):
        rng = np.random.RandomState(int(rng))
    elif rng is None:
        rng = np.random

    # Sample one array to get num_samples.
    array = list(given_arrays.values())[0]

    if not isinstance(array, np.ndarray):
        import torch

    # Apply sampling/dropout first, if necessary.
    if sample is not None and dropout is not None:
        raise ValueError("sample and dropout cannot be given at the same time")

    elif sample is not None:
        # If sample is given, convert it to dropout rate.
        original_num_samples = len(array)
        if sample <= 0 or sample > original_num_samples:
            raise ValueError("invalid value {sample} is given to sample")
        elif sample > 1:
            sample = sample / original_num_samples
        dropout = 1 - sample

    if dropout is not None and dropout > 0.0:
        if dropout >= 1.0:
            raise ValueError(f"invalid dropout rate: {dropout}")
        original_num_samples = len(array)
        target_num_samples = int(original_num_samples * (1.0 - dropout))
        filter = np.sort(
            rng.choice(
                original_num_samples, size=target_num_samples, replace=False
            )
        )
        if not isinstance(array, np.ndarray):
            filter = torch.IntTensor(filter)
        for name in given_arrays:
            given_arrays[name] = given_arrays[name][filter]

    array = list(given_arrays.values())[0]
    num_samples = len(array)

    if 0 < batch_size <= 1:
        batch_size = int(num_samples * batch_size)
        if batch_size == 0 and num_samples > 0:
            batch_size = 1
    else:
        batch_size = int(batch_size)

    if batch_size <= 0 or (batch_size > num_samples and not yield_all):
        raise ValueError("invalid batch size: %s" % str(batch_size))

    # Must be at least 1d.
    assert np.all([len(a.shape) >= 1 for a in given_arrays.values()])
    # Same sample size.
    assert np.all([len(a) == num_samples for a in given_arrays.values()])

    # shuffled_index : (num_samples,) array/tensor
    shuffled_index = rng.permutation(num_samples)
    if not isinstance(array, np.ndarray):
        shuffled_index = torch.LongTensor(shuffled_index)

    if yield_all:
        effective_num_samples = num_samples
    else:
        effective_num_samples = num_samples - (num_samples % batch_size)

    ReturnedTuple = collections.namedtuple(
        "Batch", ",".join(given_arrays.keys())
    )

    for start_index in range(0, effective_num_samples, batch_size):
        sampled_index = shuffled_index[start_index : start_index + batch_size]

        ret = {k: a[sampled_index] for k, a in given_arrays.items()}
        sampled_arrays = ReturnedTuple(**ret)
        if index:
            yield sampled_index, sampled_arrays
        else:
            yield sampled_arrays


class NotInUse(torch_nn_Module):  # type: ignore
    pass


not_in_use = NotInUse()
