# -*- coding: utf-8 -*-

"""Logger utilities

This is an utility to facilitate use of loggers.
This adds logging level to control the logger more finely
and provides helper functions to set up a logger.
"""

import copy
import logging
import logging.config
import os
import sys
import time
import typing


def initialise(filename, run_data=None) -> None:
    """Set up loggers

    This set up log handlers and sybolic links to log files..

    1) `console` handler: Output log messages to console.
    2) `infofile`, `debugfile` handlers: Save log messages to files.
       By default, the log files are named as `log/info.txt` and
       `log/debug.txt` in the current directory.
    3) `log/info_link.txt`, `log/debug_link.txt`: Symbolic links to
       the files created by `infofile`, `debugfile` handlers.

    Parameters
    ----------
    filename : str, default 'log/log.txt'
    """
    config: dict = copy.deepcopy(default_config)

    if filename is not None:
        dir = os.path.dirname(filename)
        if dir:
            os.makedirs(dir, exist_ok=True)
        for key in ["debugfile", "infofile"]:
            level = key[:-4]
            config["handlers"][key]["filename"] = add_postfix(filename, level)
    else:
        for key in ["debugfile", "infofile"]:
            del config["handlers"][key]
            config["root"]["handlers"].remove(key)
    logging.config.dictConfig(config)

    if filename is not None:
        os.makedirs("log", exist_ok=True)
        for key in ["debugfile", "infofile"]:
            symlink_path = os.path.join("log", key[:-4] + ".txt")
            if os.path.islink(symlink_path):
                os.remove(symlink_path)
            os.symlink(
                os.path.abspath(config["handlers"][key]["filename"]),
                symlink_path,
            )

    _run_data = get_run_data()
    if filename is not None:
        _run_data["info"] = config["handlers"]["infofile"]["filename"]
        _run_data["debug"] = config["handlers"]["debugfile"]["filename"]
    if run_data is not None:
        _run_data = {**_run_data, **run_data}

    logger = logging.getLogger(__name__)
    for k, v in _run_data.items():
        logger.info(f"{k}: {v}")


def add_postfix(path, postfix, sep="_"):
    """Add a postfix to a path just before the extension

    This adds a post fix to a path.  It is inserted just before the
    file extension if any, or appended at the end otherwise.

    Examples
    --------
    >>> add_postfix('foo/bar.txt', postfix='spam')
    'foo/bar_spam.txt'
    >>> add_postfix('foo/bar', postfix='spam')
    'foo/bar_spam'
    >>> add_postfix('bar', postfix='spam')
    'bar_spam'

    Parameters
    ----------
    path : str
    postfix : str

    Returns
    -------
    result : str
    """
    tmp = os.path.splitext(path)
    return tmp[0] + sep + str(postfix) + tmp[1]


_starttime = time.localtime()
starttime = time.strftime("%Y-%m-%d %H:%M:%S", _starttime)
timestamp = time.strftime("%Y%m%d_%H%M%S")


def get_run_data() -> typing.Dict[str, object]:
    """Return information on the process as a dict"""
    command: str = "python " + " ".join(sys.argv)
    hostname: str = os.uname()[1]
    pid: int = os.getpid()
    ret = {
        "command": command,
        "python": sys.executable,
        "version": sys.version_info[:3],
        "starttime": starttime,
        "host": hostname,
        "pid": pid,
    }
    return ret


default_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "plain": {"class": "logging.Formatter", "format": "%(message)s"},
        "timed": {
            "class": "logging.Formatter",
            "format": "%(asctime)s | %(message)s",
        },
        "detailed": {
            "class": "logging.Formatter",
            "format": "%(asctime)s | %(levelname)-8s | %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "stream": "ext://sys.stdout",
            "formatter": "plain",
        },
        "debugfile": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "debug.txt",
            "maxBytes": 10000000,
            "backupCount": 3,
        },
        "infofile": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": "info.txt",
            "maxBytes": 10000000,
            "backupCount": 3,
        },
    },
    "loggers": {"matplotlib": {"level": "WARNING", "propagate": False}},
    "root": {
        "level": "DEBUG",
        "handlers": [
            "console",
            "debugfile",
            "infofile",
        ],
    },
}


if __name__ == "__main__":
    import doctest

    doctest.testmod(
        optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    )


# vimquickrun: python %
