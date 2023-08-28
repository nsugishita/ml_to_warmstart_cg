# -*- coding: utf-8 -*-

"""Utilities"""

import copy
import functools
import json
import logging
import multiprocessing as mp
import numbers
import re
import time
import typing

import numpy as np

_missing: dict = {}

logger = logging.getLogger(__name__)

_set_logger = False


class DummyEvent:
    """Settable event"""

    def __init__(self, set=True):
        self._set = set

    def wait(self):
        raise ValueError("DummyEvent.wait is called")

    def set(self):
        self._set = True

    def is_set(self):
        """Return False."""
        return self._set


def yield_columns(matrix):
    """Given a sparse matrix, yield columns.

    Yields
    ------
    col_index : int
    row_index : 1d array
    data : 1d array

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>> a = sparse.coo_matrix(np.array([
    ...     [ 0, -1,  0,  1,  1],
    ...     [ 1,  0,  0,  0,  0],
    ...     [ 0,  1,  2,  0,  0],
    ...     [ 2,  0,  0,  0,  0],
    ... ]))
    >>> for col_index, row_index, data in yield_columns(a):
    ...     print(col_index)
    ...     print(row_index)
    ...     print(data)
    0
    [1 3]
    [1 2]
    1
    [0 2]
    [-1  1]
    ...
    4
    [0]
    [1]
    """
    matrix = matrix.tocsc()
    indptr = matrix.indptr
    indices = matrix.indices
    data = matrix.data
    for col_index in range(matrix.shape[1]):
        start = indptr[col_index]
        stop = indptr[col_index + 1]
        yield col_index, indices[start:stop], data[start:stop]


def format_elapse(elapse: float) -> str:
    """Given elapse in seconds, return a formatted string.

    Parameters
    ----------
    elapse : float
        E.g., a value obtained as a difference of time.perf_counter().

    Returns
    -------
    text : str
        Formatted text, which looks like 'hh:mm:ss'.
    """
    elapse_h = int(elapse // (60 * 60))
    elapse = elapse - elapse_h * 60 * 60
    elapse_m = int(elapse // 60)
    elapse_s = int(elapse - (elapse_m * 60))
    return f"{elapse_h:02d}:{elapse_m:02d}:{elapse_s:02d}"


class _ArgparseMissing(object):
    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


argparse_missing = _ArgparseMissing()

type_name_to_type = {
    "str": str,
    "int": int,
    "float": float,
    "number": float,
    "bool": bool,
}


def setup_argparser(rule: typing.Mapping, set_default_from_rule=False):
    """Return an argument parser to accept options to modify config.

    Parameters
    ----------
    rule : dict
        Rule typically passed to validdict.
    set_default_from_rule : bool, default False
        If True, set default value used in rule.

    Returns
    -------
    argparser : argparse.ArgumentParser
    mapping : dict
        typing.Mapping from the destinations of the parser
        to the config key.
    """
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    mapping = {}
    for key, value in rule.items():
        if value.get("argparse", False) is False:
            continue
        if isinstance(value["argparse"], dict):
            parser_option = value["argparse"]
        else:
            parser_option = {}
        args = ()
        kwargs = {}
        if "names" in parser_option:
            if isinstance(parser_option["names"], str):
                names = [parser_option["names"]]
            else:
                names = parser_option["names"]
        else:
            names = [key]
        for i in range(len(names)):
            names[i] = names[i].replace(".", "-").replace("_", "-")
            if not names[i].startswith("-"):
                names[i] = "--" + names[i]
        kwargs["dest"] = key.replace(".", "-").replace("_", "-")
        if "action" in parser_option:
            kwargs["action"] = parser_option["action"]
            if "default" in parser_option:
                kwargs["default"] = parser_option["default"]
        else:
            if "type" in parser_option:
                if isinstance(parser_option["type"], str):
                    kwargs["type"] = type_name_to_type[parser_option["type"]]
                else:
                    kwargs["type"] = parser_option["type"]
            elif "type" in value:
                if isinstance(value["type"], str):
                    kwargs["type"] = type_name_to_type[value["type"]]
                else:
                    kwargs["type"] = value["type"]
            if "default" in parser_option:
                kwargs["default"] = parser_option["default"]
            elif set_default_from_rule and ("default" in value):
                kwargs["default"] = value["default"]
            else:
                kwargs["default"] = argparse_missing
        args = tuple(names) + args  # type: ignore
        parser.add_argument(*args, **kwargs)
        mapping[kwargs["dest"]] = key
    return parser, mapping


def make_read_only(
    o, _memo: typing.Optional[set] = None, _level: int = 0
) -> None:
    """Find all attributes and contents read only.

    This scans all attributes and contents (if it's a container
    like a list) recursively and makes all numpy arrays to be read only.

    Examples
    --------
    >>> a = [np.arange(3), np.arange(4)]
    >>> make_read_only(a)
    >>> a[1][2] = 0
    Traceback (most recent call last):
     ...
    ValueError: assignment destination is read-only
    >>> from types import SimpleNamespace
    >>> a = SimpleNamespace()
    >>> a.data = np.empty(4)
    >>> make_read_only(a)
    >>> a.data[1] = 10
    Traceback (most recent call last):
     ...
    ValueError: assignment destination is read-only

    Parameters
    ----------
    o : obj
    """
    if _memo is None:
        _memo = set()
    if _level > 30:
        return
    if id(o) in _memo:
        return
    if isinstance(o, np.ndarray):
        # If a numpy array is given, make it read only.
        o.setflags(write=False)
    _memo.add(id(o))
    try:  # Dict like.
        for v in o.values():
            make_read_only(v, _memo=_memo, _level=_level + 1)
    except (AttributeError, TypeError):
        # AttributeError is o.values does not exist.
        # TypeError is o is type (class).
        pass
    try:
        for v in o:
            make_read_only(v, _memo=_memo, _level=_level + 1)
    except TypeError:
        pass  # TypeError if `o` is not iterable.
    for v in getattr(o, "__dict__", {}).values():
        make_read_only(v, _memo=_memo, _level=_level + 1)
    for v in getattr(o, "__slots__", {}):
        try:
            u = getattr(o, v)
        except AttributeError:
            pass
        else:
            make_read_only(u, _memo=_memo, _level=_level + 1)
    if _level == 0:
        make_read_only(o.__class__, _memo=_memo, _level=_level + 1)


def nbytes(data: typing.Mapping) -> float:
    """Return total data size of numpy arrays stored in a given dict."""
    nbytes: float = 0.0
    for key in data:
        if isinstance(data[key], np.ndarray):
            nbytes += data[key].nbytes
    return nbytes


class ProcessesContext:
    """Context to terminate processes on exit."""

    def __init__(
        self, processes: typing.Optional[typing.List[mp.Process]] = None
    ) -> None:
        """Initialise an FreezeContext instance."""
        if processes is None:
            self.processes = []
        else:
            self.processes = processes

    def append(self, process: mp.Process) -> None:
        """Append a process to manage."""
        self.processes.append(process)

    def extend(self, processes: typing.Iterable[mp.Process]) -> None:
        """Extend the process list."""
        self.processes.extend(processes)

    def is_alive(self) -> bool:
        """Returns True if any of the processes is alive."""
        return any(p.is_alive() for p in self.processes)

    def error(self) -> bool:
        """Return True if any of the processes terminated with error."""

        def terminated_with_error(p):
            return (p.exitcode is not None) and (p.exitcode != 0)

        return any(terminated_with_error(p) for p in self.processes)

    def join(self) -> None:
        """Join all processes."""
        for process in self.processes:
            process.join()

    def __enter__(self, *args, **kwargs):
        # type: (object, object) -> ProcessesContext
        """Enter the context.

        Returns
        -------
        self : ProcessesContext
        """
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Freeze the given object and exit the context."""
        for p in self.processes:
            p.terminate()
            p.join()


def accumulate(
    accumulator: typing.Mapping,
    data: typing.Mapping,
    stacked: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    concatenated: typing.Optional[typing.Union[str, typing.List[str]]] = None,
    on_error: str = "pass",
) -> None:
    """Accumulate items in `data` to `accumulator`.

    This takes all items in `data` and stack or concatenate in
    `accumulator`.

    Examples
    --------
    >>> acc = {}
    >>> data1 = {'type': 0, 'values': [2, 4, 1], 'nflags': 2, 'flags': [1, 2]}
    >>> data2 = {'type': 1, 'values': [0, 5, 4], 'nflags': 1, 'flags': [3]}
    >>> data3 = {'type': 3, 'values': [1, 1, 2], 'nflags': 2, 'flags': [4, 3]}
    >>> accumulate(acc, data1, concatenated='flags')
    >>> accumulate(acc, data2, concatenated='flags')
    >>> accumulate(acc, data3, concatenated='flags')
    >>> print(acc['type'])
    [0 1 3]
    >>> print(acc['values'])
    [[2 4 1]
     [0 5 4]
     [1 1 2]]
    >>> print(acc['flags'])
    [1 2 3 4 3]

    One can use this strictly by setting `on_error='raise'` or
    `on_error='warn'`.

    >>> accumulate(acc, data3, concatenated='flags', on_error='raise')
    Traceback (most recent call last):
     ...
    ValueError: errors in accumulating data
    unknown items ['type', 'values', 'nflags']
    >>> accumulate(
    ...     acc,
    ...     data3,
    ...     stacked=['type', 'values', 'nflags'],
    ...     concatenated='flags',
    ...     on_error='raise'
    ... )

    Parameters
    ----------
    accumulator : dict
    data : dict
    stacked : list of str, optional
    concatenated : list of str, optional
    on_error : {'pass', 'warn', 'raise'}, default 'pass'
        Specify behavior when there are unknown items
        or missing items.
        If 'pass', all unknown items are stacked.
        If 'warn', this emits warning on the logger.
        If 'raise', this raises an KeyError.
    """
    if stacked is None:
        stacked = []
    elif isinstance(stacked, str):
        stacked = [stacked]
    if concatenated is None:
        concatenated = []
    elif isinstance(concatenated, str):
        concatenated = [concatenated]
    unknown_items = []
    missing_items = list(
        set(stacked).union(set(concatenated)) - set(data.keys())
    )
    for key in data:
        if key in concatenated:
            try:
                concatenate(accumulator, key, data[key])
            except Exception as e:
                raise ValueError(
                    f"Exception raised while concatenating {key}  {str(e)}"
                )
        else:
            if key not in stacked:
                unknown_items.append(key)
            try:
                stack(accumulator, key, data[key])
            except Exception as e:
                raise ValueError(
                    f"Exception raised while stacking {key}  {str(e)}"
                )
    msg: typing.List[str] = []
    if unknown_items:
        msg.append(f"unknown items {unknown_items}")
    if missing_items:
        msg.append(f"missing items {missing_items}")
    if msg:
        joined_msg: str = "errors in accumulating data\n" + "\n".join(msg)
        if on_error == "warn":
            import logging

            logger = logging.getLogger(__name__)
            logger.warn(joined_msg)
        elif on_error == "raise":
            raise ValueError(joined_msg)


def stack(
    data, _key=_missing, _value=_missing, *, lazy_expand=False, **kwargs
) -> None:
    """Append numpy array(s).

    This appends numpy array(s).  If the key already exists,
    this stacks them.  If it is a new item and lazy_expand is
    False (default), this expands the first axis of the item
    and add them.  Otherwise, this simply puts the new array
    as a new item and the first axis is expanded when another
    array is appended in the next call.

    There are two signatures to call this method:
        stack(data, key, value)
        stack(data, key1=value1, key2=value2, ...)

    Examples
    --------
    >>> data = {}
    >>> stack(data, x=np.zeros(2))
    >>> data['x'].shape
    (1, 2)
    >>> stack(data, x=np.zeros(2))
    >>> data['x'].shape
    (2, 2)
    >>> stack(data, y=np.zeros(2), lazy_expand=True)
    >>> data['y'].shape
    (2,)
    >>> stack(data, y=np.zeros(2))
    >>> data['y'].shape
    (2, 2)

    Parameters
    ----------
    lazy_expand : bool, default False
        Expand the first axis of arrays only when it's necessary.
        If this is False, this expand the first axis of
        new arrays immediately.
    """
    if (_key is _missing) ^ (_value is _missing):
        raise ValueError(
            "must be called by stack(key, value) or stack(key=value)"
        )

    if _key is not _missing:
        _stack(data, _key=_key, _value=_value, lazy_expand=lazy_expand)

    else:
        for _key, _value in kwargs.items():
            stack(data, _key=_key, _value=_value, lazy_expand=lazy_expand)


def _stack(data, _key, _value, lazy_expand):
    if isinstance(_value, np.ndarray):
        pass
    elif isinstance(_value, (numbers.Number, str, list, tuple)):
        _value = np.array(_value)
    else:
        raise ValueError(
            "Trying to append a non numpy array: key: %s, value: %s"
            % (_key, _value.__class__.__name__)
        )
    if _key in data:
        _original_value = data[_key]
        if _original_value.shape == _value.shape:
            data[_key] = np.stack([_original_value, _value])
        elif _original_value.shape[1:] == _value.shape:
            data[_key] = np.concatenate(
                [_original_value, np.expand_dims(_value, axis=0)]
            )
        else:
            raise ValueError(
                "'%s' has invalid shape. stored: %s, got: %s"
                % (_key, _original_value.shape, _value.shape)
            )
    elif lazy_expand:
        data[_key] = _value
    else:
        data[_key] = np.expand_dims(_value, axis=0)


def concatenate(
    data, _key=_missing, _value=_missing, *, axis=0, **kwargs
) -> None:
    """Concatenate numpy array(s).

    This concatenate numpy array(s).  If the key already exists,
    this concatenates them.  If it is a new item, this simply
    puts the new array as a new item.

    There are two signatures to call this method:
        concatenate(data, key, value)
        concatenate(data, key1=value1, key2=value2, ...)

    Examples
    --------
    >>> data = {}
    >>> concatenate(data, x=np.zeros(2))
    >>> data['x'].shape
    (2,)
    >>> concatenate(data, x=np.zeros(2))
    >>> data['x'].shape
    (4,)
    >>> concatenate(data, y=np.zeros((2, 3)))
    >>> data['y'].shape
    (2, 3)
    >>> concatenate(data, y=np.zeros((3, 3)))
    >>> data['y'].shape
    (5, 3)

    Parameters
    ----------
    axis : int, default 0
        Axis on which arrays are concatenated along.
    """
    invalid_signature = _key is not _missing and _value is _missing
    invalid_signature |= _key is _missing and _value is not _missing
    if invalid_signature:
        raise ValueError(
            "must be called by concatenate(key, value) or "
            "concatenate(key=value)"
        )

    if _key is not _missing:
        _concatenate(data, _key=_key, _value=_value, axis=axis)

    else:
        for _key, _value in kwargs.items():
            _concatenate(data, _key=_key, _value=_value, axis=axis)


def _concatenate(data, _key, _value, axis):
    if isinstance(_value, np.ndarray):
        pass
    elif isinstance(_value, (numbers.Number, str, list, tuple)):
        _value = np.atleast_1d(_value)
    else:
        raise ValueError(
            "Trying to append a non numpy array: key: %s, value: %s"
            % (_key, _value.__class__.__name__)
        )
    if _value.ndim == 0:
        _value = np.atleast_1d(_value)
    if _key in data:
        _original_value = data[_key]
        try:
            data[_key] = np.concatenate([_original_value, _value], axis=axis)
        except ValueError:
            raise ValueError(
                f"invalid dimensions on key {_key}.  "
                f"stored: {_original_value.shape}  "
                f"got: {_value.shape}"
            ) from None

    else:
        data[_key] = _value


seconds_synonyms: typing.List[str] = ["s", "sec", "secs", "second", "seconds"]
minutes_synonyms: typing.List[str] = ["m", "min", "mins", "minute", "minutes"]
hours_synonyms: typing.List[str] = ["h", "hour", "hours"]
days_synonyms: typing.List[str] = ["d", "day", "days"]
weeks_synonyms: typing.List[str] = ["w", "week", "weeks"]


def parse_time_length(
    value: str,
    allow_inf: bool = True,
    default_unit: typing.Optional[str] = None,
) -> float:
    """Parse str of time length.

    This parses a string of time length into float in seconds.
    This may be useful to parse user input from command line.

    Examples
    --------
    >>> parse_time_length('1min')
    60.0
    >>> parse_time_length('0.5 hour')
    1800.0
    >>> parse_time_length('2hours30minutes')
    9000.0
    >>> parse_time_length('0m30s')
    30.0
    >>> parse_time_length('1.5days')
    129600.0
    >>> parse_time_length('inf')
    inf
    >>> parse_time_length('1')
    Traceback (most recent call last):
      ...
    ValueError: 1
    >>> parse_time_length('1', default_unit='min')
    60.0
    >>> parse_time_length('1s', default_unit='min')
    1.0

    Parameters
    ----------
    value : str
        Value to be parsed.
    allow_inf : bool, default True

    Returns
    -------
    parsed : float

    Raises
    ------
    ValueError
    """
    if value.lower() in ["inf", "infinity"]:
        if allow_inf:
            return float("infinity")
        else:
            raise ValueError(value)
    if (default_unit is not None) and re.fullmatch(
        r"[\d\-+][\d\.]*", value.strip()
    ):
        units_synonyms = (
            seconds_synonyms
            + minutes_synonyms
            + hours_synonyms
            + days_synonyms
            + weeks_synonyms
        )
        if default_unit not in units_synonyms:
            raise ValueError(f"invalid unit: {default_unit}")
        _value = value + default_unit
    else:
        _value = value
    # Split numbers and words.
    split = re.findall(r"([\d\-+][\d\.]*|[a-zA-Z\._]+)", _value)
    if len(split) % 2 != 0:
        raise ValueError(value)
    ret: float = 0.0
    for i in range(len(split) // 2):
        try:
            number = float(split[2 * i])
        except ValueError:
            raise ValueError(value) from None
        unit = split[2 * i + 1]
        if unit.lower() in seconds_synonyms:
            factor = 1
        elif unit.lower() in minutes_synonyms:
            factor = 60
        elif unit.lower() in hours_synonyms:
            factor = 60 * 60
        elif unit.lower() in days_synonyms:
            factor = 24 * 60 * 60
        elif unit.lower() in weeks_synonyms:
            factor = 7 * 24 * 60 * 60
        else:
            raise ValueError(value)
        ret += number * factor
    return ret


def parse_ints(value, numpy: bool = False, atleast_1d: bool = True):
    """Parse str of ints or range.

    This parses a string of int, ints or range.
    This may be useful to parse user input from command line.

    Examples
    --------
    >>> parse_ints('1')  # if a single number is given, it's simply parsed
    [1]
    >>> parse_ints('2, 10')  # multiple values separated by commas.
    [2, 10]
    >>> parse_ints('-3,1,-5,0')
    [-3, 1, -5, 0]
    >>> parse_ints('10:15')  # range can be specified with colons.
    range(10, 15)
    >>> parse_ints('0:10:2')  # one can specify the step
    range(0, 10, 2)
    >>> parse_ints(':2')  # if the start is omitted, 0 is used.
    range(0, 2)
    >>> parse_ints('5-20')  # dashes are interpretted as a colon.
    range(5, 20)
    >>> parse_ints('0:3.4')  # invalid values raises an ValueError
    Traceback (most recent call last):
      ...
    ValueError: 0:3.4

    Parameters
    ----------
    value : str
        Value to be parsed.
    numpy : bool, default False
        Return numpy array instead of python native objects.
    atleast_1d : bool, default True
        Return a list even if only single number is given.

    Returns
    -------
    parsed : int, list or array

    Raises
    ------
    ValueError
    """
    given = value
    ret: object
    if isinstance(value, (list, tuple)):

        def is_float(x):
            return isinstance(x, float)

        if any(is_float(x) for x in value):
            raise ValueError(given)
        ret = [int(x) for x in value]
        if numpy:
            return np.array(ret)
        else:
            return ret

    value = value.strip()
    if isinstance(value, str):
        value = re.sub(r"(\d)-(\d)", r"\g<1>:\g<2>", value)
    if value is None or value == "":
        return [] if atleast_1d else None
    if isinstance(value, (list, tuple)):
        if all([isinstance(x, int) for x in value]):
            return value
        else:
            raise ValueError(given)

    if isinstance(value, str):
        if value.startswith(":"):
            value = "0" + value

        if ":" in value:
            splitted = value.split(":")
            if any(s == "" for s in splitted):
                raise ValueError(given)
            if len(splitted) == 2:
                try:
                    start, stop = map(int, splitted)
                except ValueError:
                    pass
                else:
                    if numpy:
                        ret = np.arange(start, stop)
                    else:
                        ret = range(start, stop)
                    return ret
            elif len(splitted) == 3:
                try:
                    start, stop, step = map(int, splitted)
                except ValueError:
                    pass
                else:
                    if numpy:
                        ret = np.arange(start, stop, step)
                    else:
                        ret = range(start, stop, step)
                    return ret

        elif "," in value:
            splitted = value.split(",")
            try:
                ret = list(map(int, splitted))
            except ValueError:
                pass
            else:
                if numpy:
                    ret = np.array(ret)
                return ret

    try:
        parsed = int(value)
    except ValueError:
        pass
    else:
        if numpy and atleast_1d:
            return np.array([parsed])
        elif atleast_1d:
            return [parsed]
        else:
            return parsed

    raise ValueError(given)


class frozendict(dict):
    """Dict which can be frozen.

    Examples
    --------
    >>> a = frozendict(foo='bar', code=10)
    >>> a['foo'] = 'spam'
    >>> a['new'] = True
    Traceback (most recent call last):
     ...
    KeyError: "assigning a new item 'new' on a frozen dict"

    To assign new items, use `unfreeze`

    >>> with unfreeze(a):
    ...   a['new'] = True
    """

    __slots__ = ("_frozen",)

    def __init__(self, *args, **kwargs) -> None:
        """Initialise a ConfigDict instance."""
        super().__init__(*args, **kwargs)
        self._freeze()

    def __setitem__(self, key: str, value) -> None:
        """Set an item if it's allowed."""
        if getattr(self, "_frozen", False) and key not in self:
            raise KeyError(
                f"assigning a new item {repr(key)} on a frozen dict"
            )
        super().__setitem__(key, value)

    def __getitem__(self, key: str):
        """Get an item if it exists."""
        ret = super().__getitem__(key)
        if ret == "__missing__":
            raise KeyError(key)
        return ret

    def __getstate__(self) -> typing.Dict:
        """Return state of the dict"""
        return {
            "frozendict_data": dict(self),
            "frozendict_frozen": self._frozen,
        }

    def __setstate__(self, state: typing.Mapping) -> None:
        """Set given state"""
        # mypy does not allow to call __init__ directly.
        self.__init__(state["frozendict_data"])  # type: ignore
        self._freeze(state["frozendict_frozen"])

    def update(self, E=None, **F):
        """Update data with another dict."""
        if (E is not None) and hasattr(E, "keys"):
            for key in E.keys():
                if getattr(self, "_frozen", False) and key not in self:
                    raise KeyError(
                        f"assigning a new item {repr(key)} on a frozen dict"
                    )
        else:
            E = {}
        for key in F.keys():
            if getattr(self, "_frozen", False) and key not in self:
                raise KeyError(
                    f"assigning a new item {repr(key)} on a frozen dict"
                )
        E = {**E, **F}
        super().update(E)

    def _freeze(self, mode: bool = True) -> None:
        """Prevent new item assignments."""
        self._frozen = mode

    def _unfreeze(self):  # type: (object) -> FreezeContext
        """Allow new item assignments.

        This can be used as a context.
        """
        return unfreeze(self)


class validdict(frozendict):
    """Dict with validation.

    A simple extension of dict which validate assigned values.

    Validation rule can be given via `__init__`, `set_rule` or
    `append_rule`. A rule is expressed as a dict, whose format should be
    {
        "itemname1": {
            "required": {True|False},
            "default": obj,
            "type": {'obj'|'str'|'int'|'float'|...},  # See _check_type
            "choices": [obj, obj, ...],
            "constraint": str  # See _check_constraint
        },
        "itemname2": {
            ...
        },
    }
    For type checking, please refer to the doc of `_check_type`,
    while for constraint, that of `_check_constraint`.

    Examples
    --------
    >>> rule = {
    ...   'foo': {'type': 'int', 'constraint': 'nonnegative'},
    ...   'bar': {'type': 'bool'},
    ...   'baz': {'type': 'str', 'choices': ['a', 'b', 'c'], 'default': 'a'},
    ... }
    >>> d = validdict(rule)
    >>> d  # Default values are set.
    {'baz': 'a'}

    On can assign and retrieve items as usual, however
    each assigned values is checked against a specified type
    and/or a constraint if any.

    >>> d['foo'] = 1  # 1 is a nonnegative int and this is a valid assignment.
    >>> d['foo'] = -10  # 'foo' remains to be 1.
    Traceback (most recent call last):
     ...
    ValueError: item: foo  constraint: -10 is not nonnegative
    >>> d['baz'] = 'x'
    Traceback (most recent call last):
     ...
    ValueError: item: baz  got: x  expected: ['a', 'b', 'c']

    With `unfreeze` method, one can disable validation and
    assign new values.

    >>> d['spam'] = False
    Traceback (most recent call last):
     ...
    KeyError: "assigning a new item 'spam' on a frozen dict"
    >>> with unfreeze(d):
    ...     d['baz'] = 'x'
    ...     d['spam'] = False
    >>> d['baz']
    'x'
    >>> d['spam']
    False

    These assignment works on `update` method as well.
    Typically one can instanciate `validdict` with
    `d = validdict('rule.json')` and then set values
    from an user defined config file as `d.update(json.load(file))`.
    The value in json file is validated and if there are any
    invalid values `validdict` raises an error.

    Attributes
    ----------
    rule : dict
    validate : bool, default True
    """

    __slots__ = ("rule",)

    def __init__(
        self,
        rule: typing.Optional[typing.Union[str, typing.Mapping]] = None,
        validate: bool = True,
    ) -> None:
        """Initialise a validdict instance.

        Parameters
        ----------
        rule : dict or str, optional
        validate : bool, default True
            If True, validate each assignment.
        """
        super().__init__()
        self.rule: typing.Mapping = {}
        if rule is not None:
            self.set_rule(rule)

    def set_rule(
        self, rule: typing.Union[str, typing.Mapping], append: bool = False
    ) -> None:
        """Set a validation rule.

        This takes a dict which contains rules of items.
        Additionally, `rule` may contain `validdictconfig`
        which is a dict.  If `rule['validdictconfig']['populate_default']`
        is set to be True (by default), this populates the default
        value.

        Parameters
        ----------
        rule : dict or str
        """
        if isinstance(rule, str):
            if rule.endswith("json"):
                with open(rule, "r") as f:
                    new_rule = json.load(f)
            elif rule.endswith("yaml"):
                import yaml

                with open(rule, "r") as f:
                    new_rule = yaml.safe_load(f)
        elif isinstance(rule, dict):
            new_rule = dict(rule)
        else:
            raise ValueError("rule must be a dict or a path")
        validdict_config = new_rule.pop("validdictconfig", {})
        if append:
            self.rule = {**self.rule, **new_rule}
        else:
            self.rule = new_rule
        if validdict_config.get("populate_default", True):
            self.populate_default()

    def append_rule(self, rule: typing.Union[str, typing.Mapping]) -> None:
        """Initialise a validdict instance.

        Parameters
        ----------
        rule : dict or str
        """
        self.set_rule(rule, append=True)

    def missing(self) -> typing.List[str]:
        """Return missing items if any.

        Return
        ------
        missing : list of str
            List of keys which are missing.
        """
        ret = []
        for key in self.rule:
            required = self.rule[key].get("required", False)
            if required and (key not in self):
                ret.append(key)
        return ret

    def populate_default(self, overwrite: bool = False) -> None:
        """Set default values of missing items.

        Parameters
        ----------
        overwrite : bool, default False
            If True, overwrite existing values with the default values.
        """
        for key in self.rule:
            if "default" in self.rule[key]:
                if overwrite or (key not in self):
                    self[key] = self.rule[key]["default"]

    def __setitem__(self, key: str, value: object) -> None:
        """Set an item if it's allowed."""
        # When unpickling validdict, item may be assigned before slots are set?
        if (key in getattr(self, "rule", {})) and self._frozen:
            key, value = self._validate_key_value(key, value)
            with unfreeze(self):
                super().__setitem__(key, value)
        else:
            super().__setitem__(key, value)

    def __getstate__(self) -> typing.Dict:
        """Return state of the dict"""
        state = super().__getstate__()
        state["validdict_rule"] = self.rule
        return state

    def __setstate__(self, state: typing.Mapping) -> None:
        """Set given state"""
        # mypy does not allow to call __init__ directly.
        self._freeze(False)
        self.rule = state["validdict_rule"]
        self.update(state["frozendict_data"])
        self._freeze(state["frozendict_frozen"])

    def update(self, E=None, **F):
        """Update data with another dict."""
        if (E is not None) and hasattr(E, "keys"):
            for key in E.keys():
                self[key] = E[key]
        for key, value in F.items():
            self[key] = value

    def _validate_key_value(
        self, key: str, value: object
    ) -> typing.Tuple[str, object]:
        rule = self.rule[key]
        if "alias_to" in rule:
            mapped_key = rule["alias_to"]
            return self._validate_key_value(mapped_key, value)
        if "type" in rule:
            expected_type = rule["type"].lower()
            try:
                value = _check_type(value, expected_type)
            except TypeError:
                raise TypeError(
                    f"item: {key}  "
                    f"got: {value} ({type(value)})  "
                    f"expected: {expected_type}"
                ) from None
        if "choices" in rule:
            if value not in rule["choices"]:
                raise ValueError(
                    f"item: {key}  got: {value}  expected: {rule['choices']}"
                )
        if "constraint" in rule:
            try:
                _check_constraint(value, rule["constraint"])
            except Exception as e:
                raise ValueError(
                    f"item: {key}  constraint: {str(e)}"
                ) from None
        return key, value


def _check_type(
    value,
    expected_type: typing.Union[
        type, str, typing.List[type], typing.List[str]
    ],
) -> object:
    """Check the type of a given value.

    Currently the following types are supported:
    - 'obj'
    - 'str'
    - 'int'
    - 'float'
    - 'number'
    - 'ints'
    - 'list[one of the above item]'
    - 'multiple of the above items separated with semicolon ";"'

    Examples
    --------
    >>> _check_type('1.2', 'float')
    Traceback (most recent call last):
     ...
    TypeError
    >>> _check_type(True, bool)
    True
    >>> _check_type(1, bool)
    Traceback (most recent call last):
     ...
    TypeError
    >>> _check_type(1, 'bool;int')
    1
    >>> _check_type(1.2, 'number')
    1.2

    If list is expected, one can check the elements type as well.

    >>> _check_type([1, 2, 3], 'list[int]')
    [1, 2, 3]
    >>> _check_type([1, 2.4, 3], 'list[int]')
    Traceback (most recent call last):
     ...
    TypeError
    >>> _check_type([1, 2.4, 3], 'list')
    [1, 2.4, 3]

    Lists of multiple types are not yet supported.

    >>> _check_type([1, 2.4, 3], 'list[int;float]')
    Traceback (most recent call last):
     ...
    ValueError: invalid type: list[int;float]

    Parameters
    ----------
    value : obj
    expected_type : str, type or list of them

    Returns
    -------
    value : obj
        Returned given `value`.

    Raises
    ------
    TypeError
        Raised when invalid type is found.
    ValueError
        Raised when invalid `expected_type` is given.
    """

    def is_str(x):
        return isinstance(x, str)

    def is_type(x):
        return isinstance(x, type)

    type_list: typing.Optional[typing.List[str]] = None
    if isinstance(expected_type, str):
        type_list = expected_type.split(";")
    elif isinstance(expected_type, type):
        type_list = [expected_type.__name__]
    elif isinstance(expected_type, (list, tuple)):
        if all(is_str(x) for x in expected_type):
            expected_type = typing.cast(typing.List[str], expected_type)
            type_list = expected_type
        elif all(is_type(x) for x in expected_type):
            expected_type = typing.cast(typing.List[type], expected_type)
            type_list = [x.__name__ for x in expected_type]
    if type_list is None:
        raise ValueError(f"invalid type: {expected_type}")
    for t in type_list:
        try:
            return _check_type_impl(value=value, expected_type=t)
        except ValueError:
            raise ValueError(f"invalid type: {expected_type}") from None
        except TypeError:
            pass
    raise TypeError


def _check_type_impl(value, expected_type: str) -> object:
    # If given type is a list, check `value` type and the type of its elements.
    p = r"list(\[[a-zA-Z]+\])?"
    m = re.fullmatch(p, expected_type)
    if m:
        if m.group(1) is not None:
            element_type = m.group(1)[1:-1]
            if not isinstance(value, (list, tuple)):
                raise TypeError
        else:
            element_type = "obj"
        return [
            _check_type(value=x, expected_type=element_type) for x in value
        ]
    # Otherwise, `value` should be an object of a given type.
    if expected_type in ["obj", "object"]:
        return value
    elif expected_type in ["str", "string"]:
        return str(value)
    elif expected_type in ["bool", "boolean"]:
        # If we expect bool, don't try to parse.
        if isinstance(value, bool):
            return value
        else:
            raise TypeError
    elif expected_type == "int":
        if not isinstance(value, int):
            raise TypeError
        else:
            return value
    elif expected_type == "float":
        if not isinstance(value, float):
            raise TypeError
        else:
            return value
    elif expected_type == "number":
        if not isinstance(value, numbers.Number):
            raise TypeError
        else:
            return value
    elif expected_type == "none":
        if value is None:
            return value
        else:
            raise TypeError
    elif expected_type == "ints":

        def is_int(x):
            return isinstance(x, int)

        if isinstance(value, (list, tuple)) and all(is_int(x) for x in value):
            return value
        else:
            raise TypeError
    else:
        raise ValueError(f"invalid type: {expected_type}")


def _check_constraint(value, constraint: str) -> None:
    """Verify a given value satisfies a given constraint.

    Currently the following constraints are supported:
    - positive
    - negative
    - nonpositive
    - nonnegative
    - (x, y)
    - (x, y]
    - [x, y)
    - [x, y]
    - >= x
    - > x
    - <= x
    - < x
    - != x
    One can use semicolon ';' to use multiple constraints.

    Examples
    --------
    >>> _check_constraint(1, '>0')
    >>> _check_constraint(1, '(0, 1)')
    Traceback (most recent call last):
     ...
    ValueError: 1 is not in (0, 1)
    >>> _check_constraint(1, 'nonnegative;!=10')

    Parameters
    ----------
    value : obj
        Value to be tested.
    constraint : str
        Constraints to be met.

    Raises
    ------
    ValueError
        Raised when invalid constraint is given or it is not satisfied.
    """
    given = constraint
    constraint = constraint.strip().lower()
    if ";" in constraint:
        for constraint in constraint.split(";"):
            _check_constraint(value, constraint)
        return
    else:
        if constraint == "positive":
            if value <= 0:
                raise ValueError(f"{value} is not positive")
            return
        elif constraint == "negative":
            if value >= 0:
                raise ValueError(f"{value} is not negative")
            return
        elif constraint == "nonpositive":
            if value > 0:
                raise ValueError(f"{value} is not nonpositive")
            return
        elif constraint == "nonnegative":
            if value < 0:
                raise ValueError(f"{value} is not nonnegative")
            return
        # [xxx, xxx]
        m = re.fullmatch(
            r"(\[|\()\s*([\d\.\-+e]+),\s*([\d\.\-+e]+)\s*(\]|\))", constraint
        )
        if m:
            start_bracket = m.group(1)
            end_bracket = m.group(4)
            try:
                start = float(m.group(2))
            except ValueError:
                raise ValueError(f"invalid constraint {constraint}") from None
            try:
                end = float(m.group(3))
            except ValueError:
                raise ValueError(f"invalid constraint {constraint}") from None
            if (start_bracket == "[") and (end_bracket == "]"):
                if (value < start) or (value > end):
                    raise ValueError(f"{value} is not in {constraint}")
            elif (start_bracket == "[") and (end_bracket == ")"):
                if (value < start) or (value >= end):
                    raise ValueError(f"{value} is not in {constraint}")
            elif (start_bracket == "(") and (end_bracket == "]"):
                if (value <= start) or (value > end):
                    raise ValueError(f"{value} is not in {constraint}")
            elif (start_bracket == "(") and (end_bracket == ")"):
                if (value <= start) or (value >= end):
                    raise ValueError(f"{value} is not in {constraint}")
            return
        # <= xxx, < xxx, >= xxx, > xxx, != xxx
        m = re.fullmatch(r"(<=|>=|<|>|!=)\s*([\d\.\-+e]+)", constraint)
        if m:
            relation = m.group(1)
            try:
                rhs = float(m.group(2))
            except ValueError:
                raise ValueError(f"invalid constraint {constraint}") from None
            if relation == "<=":
                if value > rhs:
                    raise ValueError(f"{value} not {constraint}")
            elif relation == ">=":
                if value < rhs:
                    raise ValueError(f"{value} not {constraint}")
            elif relation == "<":
                if value >= rhs:
                    raise ValueError(f"{value} not {constraint}")
            elif relation == ">":
                if value <= rhs:
                    raise ValueError(f"{value} not {constraint}")
            elif relation == "!=":
                if value == rhs:
                    raise ValueError(f"{value} not {constraint}")
            return
        raise ValueError(f"invalid constraint {given}")


def freeze_attributes(cls):
    """Decorator to prevent new attribute assignments.

    A decorated class does not accept new attribute assignments
    after __init__.
    One can call 'unfreeze' to temporally allow new
    attribute and then call 'freeze_attributes' to protect again.

    This is mainly for development purposes and has many limitations.
    For example, when creating a subclass of a frozen class,
    one needs to unfreeze the attributes in the subclass' __init__.

    Examples
    --------
    >>> @freeze_attributes
    ... class A:
    ...     def __init__(self):
    ...         self.name = 'foo'

    Instances of A do not accept new attributes, while
    editing existing attributes are allowed.

    >>> a = A()
    >>> a.name = 'bar'
    >>> a.age = 10
    Traceback (most recent call last):
     ...
    AttributeError: age

    Use `unfreeze` to assign new attributes.

    >>> with unfreeze(a):
    ...     a.age = 10
    >>> a.age = 20
    """

    original_init = cls.__init__

    @functools.wraps(cls.__init__)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._frozen = True

    cls.__init__ = init

    original_setattr = cls.__setattr__

    @functools.wraps(cls.__setattr__)
    def setattr(self, key, value):
        if getattr(self, "_frozen", False) and not hasattr(self, key):
            raise AttributeError(key)
        original_setattr(self, key, value)

    cls.__setattr__ = setattr

    def _freeze(self, mode=True):
        """Prevent new item assignments."""
        self._frozen = mode

    def _unfreeze(self):
        """Allow new item assignments."""
        return unfreeze(self)

    cls._freeze = _freeze
    cls._unfreeze = _unfreeze

    return cls


class FreezeContext:
    """Context responsible to freeze an object when exiting."""

    def __init__(self, o, original_mode: bool) -> None:
        """Initialise an FreezeContext instance."""
        self.o = o
        self.original_mode = original_mode

    def __enter__(self, *args, **kwargs) -> None:
        """Enter the context."""
        pass

    def __exit__(self, *args, **kwargs) -> None:
        """Freeze the given object and exit the context."""
        self.o._freeze(mode=self.original_mode)


def freeze(o) -> FreezeContext:
    """Prevent new item assignments on a unfrozen instance or dict."""
    original_mode = o._frozen
    o._frozen = True
    return FreezeContext(o, original_mode)


def unfreeze(o) -> FreezeContext:
    """Allow new item assignments on a frozen instance or dict."""
    original_mode = o._frozen
    o._frozen = False
    return FreezeContext(o, original_mode)


class Journal:
    """Helper to gather statistics of an experiment

    This is a adapter to handle BoundsTracker, IterationJournal
    and timer at once.  One can keep track of lower and upper
    bounds with BoundsTracker, log information on iterations
    with IterationJournal (such as step size) and measure
    computational time of routines.
    """

    # General methods.

    def __init__(self, tol, stop_event):
        """Initialise a Journal instance"""
        self.stop_event = stop_event
        self.iteration = 0
        self.bounds_tracker = BoundsTracker(tol=tol, stop_event=stop_event)
        self.iteration_journal = IterationJournal()
        self.timer = timer()

    def start_hook(self):
        """Notify the start of the solver"""
        self.bounds_tracker.start_hook()
        self.iteration_journal.start_hook()
        self.timer.start_hook()

    def iteration_start_hook(self, iteration):
        """Notify the start of a new iteration"""
        self.iteration = iteration
        self.bounds_tracker.iteration_start_hook(iteration)
        self.iteration_journal.iteration_start_hook(iteration)
        self.timer.iteration_start_hook(iteration)

    def end_hook(self, *args, **kwargs):
        """Notify the end of the solver"""
        self.timer.close(*args, **kwargs)

    def dump_data(self, out=None):
        if out is None:
            out = {}
            return_result = True
        else:
            return_result = False
        self.bounds_tracker.dump_data(out=out)
        self.iteration_journal.dump_data(out=out)
        self.timer.dump_data(out=out)
        if return_result:
            return out

    # Bounds tracker.

    def set_lb(self, *args, **kwargs):
        """Update the lower bound"""
        self.bounds_tracker.set_lb(*args, **kwargs)
        self.timer.checkpoint = self.bounds_tracker.checkpoint

    def set_ub(self, *args, **kwargs):
        """Update the upper bound"""
        self.bounds_tracker.set_ub(*args, **kwargs)
        self.timer.checkpoint = self.bounds_tracker.checkpoint

    def get_gap(self, *args, **kwargs):
        """Get the suboptimality gap"""
        return self.bounds_tracker.get_gap(*args, **kwargs)

    def get_best_lb(self):
        return self.bounds_tracker.best_lb

    def get_best_ub(self):
        return self.bounds_tracker.best_ub

    def get_lb_this_iteration(self):
        return self.bounds_tracker.best_lb_this_iteration

    def get_ub_this_iteration(self):
        return self.bounds_tracker.best_ub_this_iteration

    # Iteration journal.

    def register_iteration_items(self, **kwargs):
        return self.iteration_journal.register_iteration_items(**kwargs)

    def set_iteration_items(self, **kwargs):
        return self.iteration_journal.set_iteration_items(**kwargs)

    # timer.

    def start(self, *args, **kwargs):
        self.timer.start(*args, **kwargs)

    @property
    def walltime(self) -> float:
        return self.timer.walltime

    @property
    def proctime(self) -> float:
        return self.timer.proctime


class timer:
    """Context manager to measure time easily.

    This is a context manager to measure time easily.

    Examples
    --------
    >>> with timer() as t:
    ...    pass  # Do something
    >>> _ = t.walltime  # get the length in wall time.
    >>> _ = t.proctime  # get the length in process time.

    Also, this can be used to measure iterative routines.

    >>> t = timer()  # start measuring
    >>> for i in range(3):
    ...     t.iteration = i
    ...     t.start('foo')
    ...     # do somehing
    ...     t.start('bar')
    ...     # do another thing
    >>> t.close()  # stop measuring
    >>> t.record_iteration
    [0, 0, 1, 1, 2, 2]
    >>> t.record_type  # t.record_walltime gives duration of each.
    ['foo', 'bar', 'foo', 'bar', 'foo', 'bar']
    """

    __slots__ = (
        "start_walltime",
        "start_proctime",
        "end_walltime",
        "end_proctime",
        "record_checkpoint",
        "record_iteration",
        "record_type",
        "record_walltime",
        "record_proctime",
        "checkpoint",
        "iteration",
        "current_record_checkpoint",
        "current_record_iteration",
        "current_record_type",
        "current_record_start_walltime",
        "current_record_start_proctime",
    )

    def __init__(self) -> None:
        """Initialise a time instance"""
        self.start_walltime: float = time.perf_counter()
        self.start_proctime: float = time.process_time()
        self.end_walltime: typing.Optional[float] = None
        self.end_proctime: typing.Optional[float] = None

        self.record_checkpoint: typing.List[object] = []
        self.record_iteration: typing.List[object] = []
        self.record_type: typing.List[object] = []
        self.record_walltime: typing.List[float] = []
        self.record_proctime: typing.List[float] = []

        self.checkpoint: int = 0
        self.iteration: int = 0
        self.current_record_checkpoint: int = 0
        self.current_record_iteration: int = 0
        self.current_record_type: typing.Optional[object] = None
        self.current_record_start_walltime: float = -1
        self.current_record_start_proctime: float = -1

    @property
    def walltime(self) -> float:
        if self.end_walltime is not None:
            return self.end_walltime - self.start_walltime
        else:
            return time.perf_counter() - self.start_walltime

    @property
    def proctime(self) -> float:
        if self.end_proctime is not None:
            return self.end_proctime - self.start_proctime
        else:
            return time.process_time() - self.start_proctime

    def __enter__(self, *args, **kwargs):  # type: (object, object) -> timer
        """Do nothing but return itself and enter the context"""
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Record exit time of the context"""
        self.close()

    def start_hook(self):
        """Notify the start of the solver"""
        self.reset()

    def iteration_start_hook(self, iteration):
        """Notify the start of a new iteration"""
        self.iteration = iteration

    def reset(self):
        """Set the current time as the beginning"""
        self.start_walltime = time.perf_counter()
        self.start_proctime = time.process_time()

    def close(self):
        """Finish the whole measurement"""
        self.stop(not_found_ok=True)
        if self.end_walltime is None:
            self.end_walltime = time.perf_counter()
            self.end_proctime = time.process_time()

    def start(self, record_type):
        """Start measuring a routine"""
        self.stop(not_found_ok=True)
        self.current_record_checkpoint = self.checkpoint
        self.current_record_iteration = self.iteration
        self.current_record_type = record_type
        self.current_record_start_walltime = time.perf_counter()
        self.current_record_start_proctime = time.process_time()

    def stop(self, not_found_ok=False):
        """Stop measuring a routine"""
        if (self.current_record_type is None) and (not not_found_ok):
            raise ValueError("currently not measring any routine")
        elif self.current_record_type is not None:
            self.record_checkpoint.append(self.current_record_checkpoint)
            self.record_iteration.append(self.current_record_iteration)
            self.record_type.append(self.current_record_type)
            self.record_walltime.append(
                time.perf_counter() - self.current_record_start_walltime
            )
            self.record_proctime.append(
                time.process_time() - self.current_record_start_proctime
            )
        self.current_record_checkpoint = 0
        self.current_record_iteration = 0
        self.current_record_type = None
        self.current_record_start_walltime = -1
        self.current_record_start_proctime = -1

    def dump_data(self, out=None):
        """Output data

        This outputs data on a given dict or return as a new dict.

        Parameters
        ----------
        out : dict, optional
            If given, the results are written on this dict.
            Otherwise, a new dict is returned.

        Returns
        -------
        res : dict, optional
            This is returned when `out` is not given.
        """
        if out is None:
            return_result = True
            out = {}
        else:
            return_result = False

        out["walltime"] = self.walltime
        out["proctime"] = self.proctime
        out["n_timerecords"] = len(self.record_type)
        out["timerecord_type"] = np.array(self.record_type)
        out["timerecord_checkpoint"] = np.array(self.record_checkpoint)
        out["timerecord_iteration"] = np.array(self.record_iteration)
        out["timerecord_walltime"] = np.array(self.record_walltime)
        out["timerecord_proctime"] = np.array(self.record_proctime)

        if return_result:
            return out


class BoundsTracker:
    """Logger of lower and upper bounds"""

    def __init__(self, tol, stop_event):
        """Initialise a BoundsTracker instance"""
        self.tol = tol
        self.stop_event = stop_event
        self.checkpoints_tolerance = [0.01, 0.005, 0.003, 0.0025, 0.002, 0.001]

        self.best_lb = -np.inf
        self.best_lb_data = np.nan
        self.best_lb_this_iteration = -np.inf
        self.previous_lb_value = -np.inf
        self.lb_values = []
        self.lb_times = []
        self.lb_iterations = []

        self.best_ub = np.inf
        self.best_ub_data = np.nan
        self.best_ub_this_iteration = np.inf
        self.previous_ub_value = np.inf
        self.ub_values = []
        self.ub_times = []
        self.ub_iterations = []

        # Iteration when the tolerances are achieved.
        self.checkpoints_iteration = []
        # Wall-time when the tolerances are achieved.
        self.checkpoints_walltime = []

        # Current checkpoint and iteration index.
        self.checkpoint = 0
        self.iteration = 0

        self.start_hook()

    def start_hook(self):
        """Notify the start of the solver"""
        self.starttime = time.perf_counter()

    def iteration_start_hook(self, iteration):
        """Notify the start of a new iteration"""
        self.iteration = iteration
        self.best_lb_this_iteration = -np.inf
        self.best_ub_this_iteration = np.inf

    def set_lb(self, lb, data=np.nan, duplicate_if_same=False):
        """Update the lower bound"""
        if (lb is None) or (not np.isfinite(lb)):
            return
        self.best_lb_this_iteration = max(self.best_lb_this_iteration, lb)
        if (self.previous_lb_value == lb) and (not duplicate_if_same):
            return
        elapse = time.perf_counter() - self.starttime
        self.lb_values.append(lb)
        self.lb_times.append(elapse)
        self.lb_iterations.append(self.iteration + 1)
        self.previous_lb_value = lb
        if self.best_lb < lb:
            self.best_lb = lb
            self.best_lb_data = data
            self.update_checkpoints()

    def set_ub(self, ub, data=np.nan, duplicate_if_same=False):
        """Update the upper bound"""
        if (ub is None) or (not np.isfinite(ub)):
            return
        self.best_ub_this_iteration = min(self.best_ub_this_iteration, ub)
        if (self.previous_ub_value == ub) and (not duplicate_if_same):
            return
        elapse = time.perf_counter() - self.starttime
        self.ub_values.append(ub)
        self.ub_times.append(elapse)
        self.ub_iterations.append(self.iteration + 1)
        self.previous_ub_value = ub
        if self.best_ub > ub:
            self.best_ub = ub
            self.best_ub_data = data
            self.update_checkpoints()

    def get_gap(self, lb=None, ub=None):
        """Get the suboptimality gap"""
        if lb is None:
            lb = self.best_lb
        if ub is None:
            ub = self.best_ub
        if np.isfinite(ub):
            return (ub - lb) / ub
        else:
            return np.inf

    def update_checkpoints(self):
        """Update the checkpoint and stop_event using the current gap"""
        gap = self.get_gap()
        n_checkpoints = len(self.checkpoints_tolerance)
        walltime = None
        if gap <= self.tol:
            self.stop_event.set()
        while True:
            test = (self.checkpoint < n_checkpoints) and (
                gap <= self.checkpoints_tolerance[self.checkpoint]
            )
            if test:
                if walltime is None:
                    walltime = time.perf_counter() - self.starttime
                self.checkpoints_iteration.append(self.iteration + 1)
                self.checkpoints_walltime.append(walltime)
                self.checkpoint += 1
            else:
                break

    def dump_data(self, out=None):
        """Output data

        This outputs data on a given dict or return as a new dict.

        Parameters
        ----------
        out : dict, optional
            If given, the results are written on this dict.
            Otherwise, a new dict is returned.

        Returns
        -------
        res : dict, optional
            This is returned when `out` is not given.
        """
        if out is None:
            return_result = True
            out = {}
        else:
            return_result = False

        out["best_lb"] = self.best_lb
        out["best_lb_data"] = self.best_lb_data
        out["n_all_lb_values"] = len(self.lb_values)
        out["all_lb_values"] = self.lb_values
        out["all_lb_times"] = self.lb_times
        out["all_lb_iterations"] = self.lb_iterations
        iterations = np.arange(1, self.iteration + 2)
        v, t = _get_bounds_by_iterations(
            function=max,
            value=self.lb_values,
            time=self.lb_times,
            iteration=self.lb_iterations,
            iterations=iterations,
        )
        out["n_iterations"] = len(v)
        out["iter_lb"] = v
        out["iter_lb_time"] = t

        out["best_ub"] = self.best_ub
        out["best_ub_data"] = self.best_ub_data
        out["n_all_ub_values"] = len(self.ub_values)
        out["all_ub_values"] = self.ub_values
        out["all_ub_times"] = self.ub_times
        out["all_ub_iterations"] = self.ub_iterations
        iterations = np.arange(1, self.iteration + 2)
        v, t = _get_bounds_by_iterations(
            function=min,
            value=self.ub_values,
            time=self.ub_times,
            iteration=self.ub_iterations,
            iterations=iterations,
        )
        out["iter_ub"] = v
        out["iter_ub_time"] = t

        out["checkpoints_tol"] = self.checkpoints_tolerance
        n = len(self.checkpoints_tolerance) - len(self.checkpoints_iteration)
        out["checkpoints_iteration"] = np.r_[
            self.checkpoints_iteration, [-1] * n
        ]
        out["checkpoints_walltime"] = np.r_[
            self.checkpoints_walltime, [-1] * n
        ]

        if return_result:
            return out


def _get_index_to_reduce_by_group(
    function,
    value,
    group,
    groups,
    missing=-1,
):
    """Return indexes when a given reduction is applied by groups"""
    if function is max:
        function = np.max
    elif function is min:
        function = np.min
    value = np.asarray(value)
    res = np.full(groups.shape, missing)
    for gi, g in enumerate(groups):
        selector = np.nonzero(group == g)[0]
        if len(selector) == 0:
            continue
        value_in_group = value[selector]
        chosen = function(value_in_group)
        if not np.isscalar(chosen):
            raise ValueError(
                f"given function returned multiple items {chosen} "
                "in group {g}: {value_in_group}"
            )
        index_within_group = np.nonzero(chosen == value_in_group)[0][0]
        res[gi] = selector[index_within_group]
    return res


def _get_bounds_by_iterations(function, value, time, iteration, iterations):
    index = _get_index_to_reduce_by_group(
        function=function,
        value=value,
        group=iteration,
        groups=iterations,
    )
    return (
        np.r_[value, np.nan][index],
        np.r_[time, np.nan][index],
    )


class IterationJournal:
    """Logger to save statistics bound to iterations

    This is a logger to save information related to iterations,
    such as step size.  This can save not only scalars but also
    arrays.  This keeps data as lists internally and saves values
    with the time they get available.

    Items stored in this instance must be registered with
    `register_iteration_items` method first.
    """

    def __init__(self):
        """Initialise an IterationJournal instance"""
        self.iteration_item_values = dict()
        self.iteration_item_times = dict()
        self.iteration_item_default_values = dict()
        self.iteration_item_with_timing = []
        self.start_hook()

    def start_hook(self):
        """Notify the start of the solver"""
        self.starttime = time.perf_counter()
        self.iteration = 0

    def iteration_start_hook(self, iteration):
        """Notify the start of a new iteration"""
        self.iteration = iteration
        to = self.iteration + 1
        for key in self.iteration_item_default_values:
            self._append_item(key, to=to)

    def _append_item(self, key, value=_missing, time=np.nan, n=None, to=None):
        """Extend the list by appending items

        Parameters
        ----------
        key : str
        value : object, optional
            If omitted, the default value is used.
        time : float, default np.nan
        n : int, optional
            If given, this specifies the number of items to be appended.
            If `n` and `to` is omitted, only one item is appended.
        to : int, optional
            If given, this specifies the expected length of the list
            ater appending items.  If `n` is given, this is ignored.
        """
        if value is _missing:
            value = self.iteration_item_default_values[key]
        self.iteration_item_values.setdefault(key, [])
        if key in self.iteration_item_with_timing:
            self.iteration_item_times.setdefault(key, [])
        if n is None:
            if to is None:
                n = 1
            else:
                n = to - len(self.iteration_item_values[key])
        if key in self.iteration_item_with_timing:
            for i in range(n):
                self.iteration_item_values[key].append(copy.deepcopy(value))
                self.iteration_item_times[key].append(time)
        else:
            for i in range(n):
                self.iteration_item_values[key].append(copy.deepcopy(value))

    def register_iteration_items(self, **kwargs):
        """Register items to be logged

        This registers items to be logged.  This method should
        be called with the following signature:

        ```
        register_iteration_items(
            item_name1=default_value1,
            item_name2=default_value2,
            ...
        )
        ```

        To control the behaviour finely, one can pass a dict.
        It may contain the following keys:
        - default (required) : object
            Default value
        - timing : bool, default True
            Keep timing or not.
        """
        for key, value in kwargs.items():
            if not isinstance(value, dict):
                value = dict(default=value)
            assert "default" in value
            self.iteration_item_default_values[key] = value["default"]
            if value.get("timing", True):
                self.iteration_item_with_timing.append(key)

    def set_iteration_items(self, **kwargs):
        """Set values of items

        This saves values of given items.  This method should
        be called with the following signature:

        ```
        set_iteration_items(
            item_name1=saved_value1,
            item_name2=saved_value2,
            ...
        )
        ```
        """
        it = self.iteration
        to = it + 1
        elapse = time.perf_counter() - self.starttime
        for key, value in kwargs.items():
            self._append_item(key, to=to)
            self.iteration_item_values[key][self.iteration] = value
            if key in self.iteration_item_with_timing:
                self.iteration_item_times[key][self.iteration] = elapse

    def is_iteration_item(self, key):
        return key in self.iteration_item_default_values

    def get_iteration_item(
        self,
        key,
        default=_missing,
        iteration=_missing,
        return_default_on_index_error=False,
    ):
        if not self.is_iteration_item(key):
            if default is _missing:
                raise KeyError(key)
            else:
                return default
        self._append_item(key, to=self.iteration + 1)
        if iteration is _missing:
            iteration = -1
        try:
            return self.iteration_item_values[key][iteration]
        except IndexError as e:
            if return_default_on_index_error:
                return self.iteration_item_default_values[key]
            else:
                raise e from None

    def add_iteration_item(self, key, value):
        self._append_item(key, to=self.iteration + 1)
        self.iteration_item_values[key][-1] += value

    def dump_data(
        self,
        out=None,
        value_format="iter_{key}",
        time_format="iter_{key}_time",
    ):
        """Output all saved data

        Parameters
        ----------
        out : dict, optional
            If given, the results are written on this dict.
            Otherwise, a new dict is returned.
        value_format : str, default "iter_{key}"
            Format string to define an item name in the output.
        time_format : str, default "iter_{key}_time"
            Format string to define an item name in the output.

        Returns
        -------
        res : dict, optional
            This is returned when `out` is not given.
        """
        if out is None:
            return_result = True
            out = {}
        else:
            return_result = False
        for key, value in self.iteration_item_values.items():
            out[value_format.format(key=key)] = value
            if key in self.iteration_item_with_timing:
                out[time_format.format(key=key)] = self.iteration_item_times[
                    key
                ]
        if return_result:
            return out


if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS)
