# -*- coding: utf-8 -*-

"""Utilities on cplex.

This contains some utilities to handle cplex instances,
such as a picklable extension of cplex.Cplex class.
"""

import collections
import datetime
import logging
import sys
import time
import typing

import cplex
import numpy as np

# Expose frequently used objects for convenience.
CplexError = cplex.CplexError
SparsePair = cplex.SparsePair
SparseTriple = cplex.SparseTriple
callbacks = cplex.callbacks
exceptions = cplex.exceptions
infinity = cplex.infinity

# This will be set when get_state is called for the first time.
default_cplex_attributes: typing.List[str] = []


class Cplex(cplex.Cplex):
    """Picklable Cplex.

    This is a pickable extension of Cplex class.
    Note that one can only pickle the data of the model (and parameter
    values), but not the state of the solver nor callbacks.
    See the doc of `get_state` for more details.

    Examples
    --------
    >>> import pickle
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> m = Cplex()
    >>> _ = m.set_warning_stream(None)
    >>> m.parameters.mip.tolerances.mipgap.set(0.02)
    >>> m.variables.add(lb=[-1, 2, 1])
    range(0, 3)
    >>> m.foo = 'bar'
    >>> pickle.dump(m, outfile)

    >>> _ = outfile.seek(0)  # emerate opening file.
    >>> n = pickle.load(outfile)
    >>> n.foo
    'bar'
    >>> n.parameters.mip.tolerances.mipgap.get()
    0.02
    >>> n.variables.get_lower_bounds()
    [-1.0, 2.0, 1.0]
    """

    def __getstate__(self) -> typing.Mapping:
        """Return state of the model"""
        # If this method is called for the first time, set the default
        # attribute list of cplex.
        if len(default_cplex_attributes) == 0:
            default_cplex_attributes.extend(
                list(cplex.Cplex().__dict__.keys())
            )

        state = dict(self.__dict__)

        for attribute in default_cplex_attributes:
            state.pop(attribute, None)

        state = {"__cplexattributes__": state}

        filetype = "sav"
        state["__cplexfiletype__"] = filetype
        state["__cplexproblem__"] = self.write_as_string(filetype=filetype)

        state["__cplexparameters__"] = get_changed_cplex_parameters(self)

        state["__cplexutilspickleversion__"] = 2

        return state

    def __setstate__(self, state: typing.Mapping) -> None:
        """Set given state"""
        import tempfile

        # Cplex set up self.variables etc. in the __init__ method.
        # If one extends cplex.Cplex, it is not desired to call
        # __init__ method of the extended class, so we are more
        # explicit here.
        cplex.Cplex.__init__(self)

        mode = "w+b" if isinstance(state["__cplexproblem__"], bytes) else "w"
        f = tempfile.NamedTemporaryFile(mode)
        with f:
            f.write(state["__cplexproblem__"])
            f.seek(0)
            self.read(f.name, filetype=state["__cplexfiletype__"])
        self.__dict__.update(state["__cplexattributes__"])
        set_cplex_parameters(self, state["__cplexparameters__"])


def get_changed_cplex_parameters(
    model: typing.Any,
) -> typing.List[typing.Tuple[typing.List[str], object]]:
    """Return modified parameters.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.parameters.lpmethod.set(m.parameters.lpmethod.values.dual)
    >>> m.parameters.mip.tolerances.mipgap.set(1e-2)
    >>> get_changed_cplex_parameters(m)
    [(['lpmethod'], 2), (['mip', 'tolerances', 'mipgap'], 0.01)]

    Parameters
    ----------
    model : cplex.Cplex

    Returns
    -------
    params : list
        This is a list of (param, value) pairs, where param is a list of str.
    """
    result = []
    for param, value in model.parameters.get_changed():
        name = str(param).split(".")[1:]
        result.append((name, value))
    return result


def set_cplex_parameters(
    model: typing.Any,
    params: typing.List[typing.Tuple[typing.List[str], object]],
) -> None:
    """Given a list of parameter, value pairs, set cplex parameters.

    This update the parameters of the given cplex model inplace.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.parameters.lpmethod.set(m.parameters.lpmethod.values.dual)
    >>> m.parameters.mip.tolerances.mipgap.set(1e-2)
    >>> params = get_changed_cplex_parameters(m)
    >>> n = cplex.Cplex()
    >>> n.parameters.mip.tolerances.mipgap.get()
    0.0001
    >>> set_cplex_parameters(n, params)
    >>> n.parameters.mip.tolerances.mipgap.get()
    0.01

    Parameters
    ----------
    model : cplex.Cplex
    params : list
        This should be a list of (param, value) pairs,
        where param is a list of str.
    """
    for name, value in params:
        cursor = model.parameters
        for n in name:
            cursor = getattr(cursor, n)
        cursor.set(value)


# A context manager responsible to close the stream if necessary.
class StreamCloserContext(object):
    """A context to close a stream if given."""

    def __init__(
        self,
        model: typing.Any = None,
        stream: typing.Optional[typing.IO] = None,
    ) -> None:
        """Initialise a StreamCloserContext instance."""
        self.model = model
        self.stream = stream

    def __enter__(self, *args, **kwargs) -> None:
        """Do nothing and enter the context."""
        pass

    def __exit__(self, *args, **kwargs) -> None:
        """Close a stream if any."""
        if self.stream is not None:
            self.stream.close()
            set_stream(self.model, None)


def get_prefix(prefix: typing.Optional[str] = None):
    """Create a function to return a prefix

    Parameters
    ----------
    prefix : {'timestamp', 'elapse', 'walltime', 'proctime', 'all'}, optional
        Add time stamp in the beginning of each line.
        This is done before records are passed to the hook.
        elapse is a synonym for walltime.
        If 'all' is given, 'timestamp' and 'elapse' are added.

    Returns
    -------
    prefix : callable
    """
    starttime = None

    if prefix == "timestamp":

        def impl():
            return datetime.datetime.now().isoformat() + " "

    elif prefix in ["walltime", "elapse"]:

        def impl():
            nonlocal starttime
            if starttime is None:
                starttime = time.perf_counter()
                duration = 0
            else:
                duration = time.perf_counter() - starttime
            return f"{duration:8.2f} "

    elif prefix == "proctime":

        def impl():
            nonlocal starttime
            if starttime is None:
                starttime = time.proctime()
                duration = 0
            else:
                duration = time.proctime() - starttime
            return f"{duration:8.2f} "

    elif prefix == "all":

        def impl():
            nonlocal starttime
            if starttime is None:
                starttime = time.perf_counter()
                duration = 0
            else:
                duration = time.perf_counter() - starttime
            return datetime.datetime.now().isoformat() + f" {duration:8.2f} "

    else:

        def impl():
            return ""

    return impl


def get_formatter(prefix: typing.Optional[str] = None):
    """Create a function to format a string

    Parameters
    ----------
    prefix : {'timestamp', 'elapse', 'walltime', 'proctime', 'all'}, optional
        Add time stamp in the beginning of each line.
        This is done before records are passed to the hook.
        elapse is a synonym for walltime.
        If 'all' is given, 'timestamp' and 'elapse' are added.

    Returns
    -------
    formatter : callable
    """
    prefix = get_prefix(prefix=prefix)
    add_prefix_flag = True

    def format(x):
        nonlocal add_prefix_flag
        pref = prefix()
        if add_prefix_flag:
            x = pref + x
        if x[-1] == "\n":
            add_prefix_flag = True
            x = x[:-1]
            end = "\n"
        else:
            add_prefix_flag = False
            end = ""
        x = x.replace("\n", f"\n{pref} ") + end
        return x

    return format


class LoggerAsFile(object):
    """Wrapper of a logger instance with a file-like interface

    Attributes
    ----------
    logger : logging.Logger
    level : int, default logging.INFO
        Log level to write log messages
    """

    def __init__(self, logger, level=logging.INFO):
        """Initialise a LoggerAsFile instance"""
        self.logger = logger
        self.level = level

    def write(self, value):
        """Write a log message

        Parameters
        ---------
        value : str
        """
        if value[-1] == "\n":
            value = value[:-1]
        for v in value.split("\n"):
            self.logger.log(self.level, v)

    def flush(self, *args, **kwargs):
        """Flush the content which has no effect on this object

        This method does not do anything. This exists just to be
        compatible with file-like object.
        """
        pass


def set_stream(
    model: typing.Any,
    stream: typing.Union[str, typing.IO, logging.Logger, None],
    mode: str = "a",
    level: int = logging.INFO,
    hook: typing.Callable[[str], str] = None,
    prefix: typing.Optional[str] = None,
) -> StreamCloserContext:
    """Set output streams.

    This sets the stream to write the output from the solver.
    By using as a contex, one can temporally set a stream.

    This also has an ability to add timestamps at each line of the log.
    To enable this functionality, set `prefix` to be 'timestamp'.

    Examples
    --------
    >>> from tempfile import NamedTemporaryFile
    >>> outfile = NamedTemporaryFile()
    >>> m = cplex.Cplex()
    >>> with set_stream(m, outfile.name, prefix='timestamp'):
    ...     m.solve()  # All logging will be saved in outfile.

    Parameters
    ----------
    model : cplex.Cplex
    stream : {file-like|str|logging.Logger object, 'stdout', 'stderr' or None}
        If this is str, this must be path to a file.
    mode : {'a', 'w'}, default 'a'
        Mode used when opening a file. This is only used when
        `stream` is str.
    level : int, default logging.INFO
        Level of log messages to be written. This is only used when `stream`
        is a logging.Logger instance.
    hook : callable, optional
        A hook to preprocess log records.
        If given, this should acept a str and output a str.
        If `prefix` is set, first prefix is
        appended and then this hook is called on the resulting log.
    prefix : {'timestamp', 'elapse', 'walltime', 'proctime', 'all'}, optional
        Add time stamp in the beginning of each line.
        This is done before records are passed to the hook.
        elapse is a synonym for walltime.
        If 'all' is given, 'timestamp' and 'elapse' are added.

    Returns
    -------
    context : Context Manger
        A context manager which closes a stream opened by
        this methods if any.
        If this method is not called in with statement,
        this does not do anything and it is user's
        responsibility to close the stream.
    """
    opened_stream = False

    assert prefix in [
        None,
        "timestamp",
        "elapse",
        "walltime",
        "proctime",
        "all",
    ]

    _stream: typing.Union[str, typing.IO, LoggerAsFile, None] = None
    if not stream:  # If None or an empty string, disable streams and exit.
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
        return StreamCloserContext()  # Return an empty context.
    elif isinstance(stream, logging.Logger):
        _stream = LoggerAsFile(stream, level)
    elif str(stream).lower() == "stdout":
        _stream = sys.stdout
    elif str(stream).lower() == "stderr":
        _stream = sys.stderr
    elif isinstance(stream, str):
        _stream = open(stream, mode)
        opened_stream = True
    else:
        _stream = None

    # Flag used in add_prefix
    # Cplex outputs lines in a quite unique way and each call of
    # hook does not correspond to a line (sometimes, cplex
    # outputs a line break at the beginning and puts the following content).
    # We have to keep track when line breaks happen so we can insert
    # the prefix in appropriate timings.
    add_prefix_flag = True
    # The time when the first log is emitted.
    starttime = None

    if prefix == "timestamp":

        def get_prefix():
            return datetime.datetime.now().isoformat() + " "

    elif prefix in ["walltime", "elapse"]:

        def get_prefix():
            nonlocal starttime
            if starttime is None:
                starttime = time.perf_counter()
                duration = 0
            else:
                duration = time.perf_counter() - starttime
            return f"{duration:8.2f} "

    elif prefix == "proctime":

        def get_prefix():
            nonlocal starttime
            if starttime is None:
                starttime = time.proctime()
                duration = 0
            else:
                duration = time.proctime() - starttime
            return f"{duration:8.2f} "

    elif prefix == "all":

        def get_prefix():
            nonlocal starttime
            if starttime is None:
                starttime = time.perf_counter()
                duration = 0
            else:
                duration = time.perf_counter() - starttime
            return datetime.datetime.now().isoformat() + f" {duration:8.2f} "

    def add_prefix(x):
        nonlocal add_prefix_flag
        pref = get_prefix()
        if add_prefix_flag:
            x = pref + x
        if x[-1] == "\n":
            add_prefix_flag = True
            x = x[:-1]
            end = "\n"
        else:
            add_prefix_flag = False
            end = ""
        x = x.replace("\n", f"\n{pref} ") + end
        return x

    # If a hook is given and prefix is required, we need to combine them.
    # Otherwise, set a hook appropriately.
    if (hook is not None) and (prefix is not None):
        original_hook = hook

        def hook(x):
            return original_hook(add_prefix(x))

    elif hook is None and (prefix is not None):
        hook = add_prefix
    else:
        hook = None

    model.set_log_stream(_stream, fn=hook)
    model.set_error_stream(_stream, fn=hook)
    model.set_warning_stream(_stream, fn=hook)
    model.set_results_stream(_stream, fn=hook)

    if opened_stream:
        _stream = typing.cast(typing.IO, _stream)
        return StreamCloserContext(model=model, stream=_stream)
    else:
        return StreamCloserContext()  # Return an empty context.


# A context manager responsible to un-fix variable values.
class UnfixVariablesContext(object):
    """A context to un-fix variable values if necessary."""

    def __init__(self, model: typing.Any, variables, lb, ub) -> None:
        """Initialise a UnfixVariablesContext instance."""
        self.model = model
        self.variables = variables
        self.lb = lb
        self.ub = ub

    def __enter__(self, *args, **kwargs) -> None:
        """Do nothing and enter the context."""
        pass

    def __exit__(self, *args, **kwargs) -> None:
        """Unfix variables."""
        if self.variables.size > 0:
            self.model.variables.set_lower_bounds(
                zip(
                    map(int, self.variables.ravel()),
                    map(float, self.lb.ravel()),
                )
            )
            self.model.variables.set_upper_bounds(
                zip(
                    map(int, self.variables.ravel()),
                    map(float, self.ub.ravel()),
                )
            )


def fix_variables(
    model: typing.Any,
    variables,
    values,
    validate: bool = True,
    context: bool = True,
) -> typing.Optional[UnfixVariablesContext]:
    """Fix values of specified variables.

    This fixes values of specified variables.
    If this is used as a context manager, the fixed variables
    are restored on the exit of the context.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.variables.add(lb=[-1, 4.5, 2, 5])
    range(0, 4)
    >>> with fix_variables(m, variables=[1, 2], values=[10, 20]):
    ...     m.variables.get_lower_bounds()
    [-1.0, 10.0, 20.0, 5.0]
    >>> m.variables.get_lower_bounds()
    [-1.0, 4.5, 2.0, 5.0]
    >>> fix_variables(m, variables=[1, 2], values=[5, 0])
    Traceback (most recent call last):
      ...
    AssertionError: lb is violated.
    index: [1]
    lb: [2.]
    got: [0]
    >>> with fix_variables(m, variables=[], values=[]):
    ...     pass  # Passing empty data does not modify the model.

    Parameters
    ----------
    model : cplex.Cplex
    variables : array or scalar
    values : array or scalar
    validate : bool, default True
        If True, this checkes whether given values are within
        the original bounds.
    context : bool, default True
        If False, this does not set up any context, which may
        improve performance slightly.

    Returns
    -------
    context : Context Manger, optional
        A context manager which unfix the variables fixed by this method.
        If this method is not called in with statement,
        the fixed values are kept fixed and it is user's
        responsibility to unfix the variables.
        If `context` argument is set to be False, this is not returned.
    """
    variables, values = np.broadcast_arrays(variables, values)
    if variables.size == values.size == 0:  # It's empty, return now.
        if context:
            return UnfixVariablesContext(
                model=model, variables=variables, lb=values, ub=values
            )
        else:
            return None
    if validate or context:
        # get the original lb and ub.
        original_lb = np.array(model.variables.get_lower_bounds())[variables]
        original_ub = np.array(model.variables.get_upper_bounds())[variables]
    if validate:
        # Assert given values are actually within the original bounds.
        epsilon = 1e-6
        if not np.all(values >= original_lb - epsilon):
            violated = np.nonzero(values < original_lb - epsilon)[0]
            raise AssertionError(
                f"lb is violated.\n"
                f"index: {violated}\n"
                f"lb: {original_lb[violated]}\n"
                f"got: {values[violated]}"
            )
        if not np.all(values <= original_ub + epsilon):
            violated = np.nonzero(values > original_ub + epsilon)[0]
            raise AssertionError(
                f"ub is violated.\n"
                f"index: {violated}\n"
                f"ub: {original_ub[violated]}\n"
                f"got: {values[violated]}"
            )
    model.variables.set_lower_bounds(
        zip(map(int, variables.ravel()), map(float, values.ravel()))
    )
    model.variables.set_upper_bounds(
        zip(map(int, variables.ravel()), map(float, values.ravel()))
    )
    if context:
        return UnfixVariablesContext(
            model=model,
            variables=variables,
            lb=original_lb,
            ub=original_ub,
        )
    else:
        return None


def add_slacks(model: typing.Any, constraints, penalty):
    """Add slack variables on a specified constraints.

    If the specified constraints are inequality ones with '<=' ('L') signs
    this adds slack variables 's' with coefficient -1.
    If the specified constraints are inequality ones with '>=' ('G') signs
    this adds slack variables 's' with coefficient 1.
    If the specified constraints are equality ones this raises ValueError.

    Examples
    --------
    >>> m = cplex.Cplex()
    >>> m.linear_constraints.add(senses=['L', 'G', 'E'])
    range(0, 3)
    >>> add_slacks(m, [0, 1], penalty=10)
    array([0, 1])
    >>> m.linear_constraints.get_coefficients([(0, 0), (0, 1)])
    [-1.0, 0.0]
    >>> m.linear_constraints.get_coefficients([(1, 0), (1, 1)])
    [0.0, 1.0]
    >>> m.objective.get_linear()
    [10.0, 10.0]

    Parameters
    ----------
    model : cplex.Cplex
    constraints : array of int
    penalty : array of float or scalar

    Returns
    -------
    positive_slack_indices : array of int
        The indices of added slack variables.
    """
    constraints, penalty = np.broadcast_arrays(constraints, penalty)
    senses = np.array(model.linear_constraints.get_senses())[constraints]
    lb = np.zeros(penalty.shape, dtype=float)
    ub = np.full_like(penalty, np.inf, dtype=float)
    if np.any(senses == "E"):
        raise NotImplementedError
    columns = [
        [[int(i)], [1.0 if s == "G" else -1.0]]
        for i, s in zip(constraints.ravel(), senses.ravel())
    ]
    slack_indices = model.variables.add(
        obj=penalty.ravel().tolist(),
        lb=lb.ravel().tolist(),
        ub=ub.ravel().tolist(),
        columns=columns,
    )
    slack_indices = np.array(slack_indices).reshape(constraints.shape)
    return slack_indices


def abort_at(
    model: typing.Any,
    val: float,
    callback: str = "mip_info_callback",
) -> None:
    """Set a threshold to abort the execution of a given solver.

    Parameters
    ----------
    model : cplex.Cplex
    val : float
    """
    model._abort_at = val

    assert callback in {"mip_info_callback", "incumbent_callback"}

    if callback == "mip_info_callback":
        callback_cls = cplex.callbacks.MIPInfoCallback
    elif callback == "incumbent_callback":
        callback_cls = cplex.callbacks.IncumbentCallback
    else:
        raise ValueError(callback)

    if getattr(model, "_abort_callback", None) is None:

        class CB(callback_cls):  # type: ignore
            def __call__(self):
                if self.get_incumbent_objective_value() <= model._abort_at:
                    self.abort()

        model._abort_callback = model.register_callback(CB)


AbortPredicateInfo = collections.namedtuple(
    "AbortPredicateInfo",
    [
        "cb",
        "incumbent_objective_value",
        "best_objective_value",
        "incumbent_objective_value_updated",
        "best_objective_value_update",
    ],
)
AbortPredicateInfo.__doc__ += ": Data passed to a predicate in `abort_if`"
AbortPredicateInfo.cb.__doc__ = "the callback object"
AbortPredicateInfo.incumbent_objective_value.__doc__ = (
    "the objective value of the incumbent solution"
)
AbortPredicateInfo.best_objective_value.__doc__ = (
    "the best objective value among unexplored nodes"
)


def abort_if(
    model: typing.Any,
    predicate: typing.Callable[[AbortPredicateInfo], bool],
    callback: str = "mip_info_callback",
) -> None:
    """Set a callback to decide whether to abort the solver or not.

    Parameters
    ----------
    model : cplex.Cplex
    predicate : Callable
        This is a callable with signature
        `predicate(AbortPredicateInfo) -> bool`.
        If the predicate returns True, the solver is terminated.
    callback : {"mip_info_callback" (default), "incumbent_callback"}
        Type of callback to be registered.
    """
    assert callback in {"mip_info_callback", "incumbent_callback"}

    if callback == "mip_info_callback":
        callback_cls = cplex.callbacks.MIPInfoCallback
    elif callback == "incumbent_callback":
        callback_cls = cplex.callbacks.IncumbentCallback
    else:
        raise ValueError(callback)

    # It is not clear how to annotate `callback_cls`
    # so that it is recognized as a valid class.
    # Below seems to mention this issue but the solution
    # is still unclear.
    # https://mypy.readthedocs.io/en/latest/common_issues.html
    class CB(callback_cls):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_incumbent_objective_value = np.nan
            self.prev_best_objective_value = np.nan

        def __call__(self):
            has_incumbent = self.has_incumbent()
            if has_incumbent:
                incumbent_objective_value = (
                    self.get_incumbent_objective_value()
                )
            else:
                incumbent_objective_value = np.nan
            best_objective_value = self.get_best_objective_value()
            incumbent_objective_value_updated = has_incumbent and (
                incumbent_objective_value
                != self.prev_incumbent_objective_value
            )
            best_objective_value_update = np.isfinite(
                best_objective_value
            ) and (best_objective_value != self.prev_best_objective_value)
            test = predicate(
                AbortPredicateInfo(
                    cb=self,
                    incumbent_objective_value=incumbent_objective_value,
                    best_objective_value=best_objective_value,
                    incumbent_objective_value_updated=(
                        incumbent_objective_value_updated
                    ),
                    best_objective_value_update=(best_objective_value_update),
                )
            )
            self.prev_incumbent_objective_value = incumbent_objective_value
            self.prev_best_objective_value = best_objective_value
            if test:
                self.abort()

    model.register_callback(CB)


class BoundLogger:
    """Data structure to keep data on lower and upper bounds"""

    def __init__(self):
        """Initializse a BoundLogger instance"""
        self.walltime_start = time.perf_counter()
        self.proctime_start = time.process_time()
        self.lb_values = []
        self.lb_times = []
        self.ub_values = []
        self.ub_times = []

    @property
    def walltime(self):
        return time.perf_counter() - self.walltime_start

    @property
    def proctime(self):
        return time.process_time() - self.proctime_start


def set_bound_logger(
    model: typing.Any,
    callback: str = "mip_info_callback",
) -> BoundLogger:
    """Set a callback to decide whether to abort the solver or not.

    Parameters
    ----------
    model : cplex.Cplex
    predicate : Callable
        This is a callable with signature
        `predicate(AbortPredicateInfo) -> bool`.
        If the predicate returns True, the solver is terminated.
    callback : {"mip_info_callback" (default), "incumbent_callback"}
        Type of callback to be registered.
    """
    assert callback in {"mip_info_callback", "incumbent_callback"}

    if callback == "mip_info_callback":
        callback_cls = cplex.callbacks.MIPInfoCallback
    elif callback == "incumbent_callback":
        callback_cls = cplex.callbacks.IncumbentCallback
    else:
        raise ValueError(callback)

    bound_logger = BoundLogger()
    lb_values = bound_logger.lb_values
    lb_times = bound_logger.lb_times
    ub_values = bound_logger.ub_values
    ub_times = bound_logger.ub_times

    # See the above comments on typing `callback_cls`.
    class CB(callback_cls):  # type: ignore
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_incumbent_objective_value = np.inf
            self.prev_best_objective_value = -np.inf

        def __call__(self):
            if self.has_incumbent():
                incumbent_objective_value = (
                    self.get_incumbent_objective_value()
                )
                if (
                    incumbent_objective_value
                    < self.prev_incumbent_objective_value
                ):
                    self.prev_incumbent_objective_value = (
                        incumbent_objective_value
                    )
                    ub_values.append(incumbent_objective_value)
                    ub_times.append(
                        time.perf_counter() - bound_logger.walltime_start
                    )
            best_objective_value = self.get_best_objective_value()
            if best_objective_value > self.prev_best_objective_value:
                self.prev_best_objective_value = best_objective_value
                lb_values.append(best_objective_value)
                lb_times.append(
                    time.perf_counter() - bound_logger.walltime_start
                )

    model.register_callback(CB)

    return bound_logger


def is_mip_optimal(model, allow_abort=False, allow_limit=False):
    """Check whether a solution is found or not.

    Parameters
    ----------
    model
    allow_abort : bool, default False
        If True, this allows the model to be aborted.
    allow_limit : bool, default False
        If True, this allows the model to be terminated
        due to time or solution limit.

    Returns
    -------
    ret : bool
    """
    # 101 CPXMIP_OPTIMAL
    # 102 CPXMIP_OPTIMAL_TOL
    # 113 CPXMIP_ABORT_FEAS
    # 104 CPXMIP_SOL_LIM
    # 107 CPXMIP_TIME_LIM_FEAS
    status = model.solution.get_status()
    if (status == 101) or (status == 102):
        return True
    if allow_abort and (status == 113):
        return True
    if allow_limit and ((status == 104) or (status == 107)):
        return True
    return False


def is_mip_feasible(model):
    """Check whether a solution is found or not.

    Parameters
    ----------
    model

    Returns
    -------
    ret : bool
    """
    # 101 CPXMIP_OPTIMAL
    # 102 CPXMIP_OPTIMAL_TOL
    # 113 CPXMIP_ABORT_FEAS
    # 104 CPXMIP_SOL_LIM
    # 107 CPXMIP_TIME_LIM_FEAS
    status = model.solution.get_status()
    return status in [101, 102, 104, 107, 113]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
