import functools
import os
import sys
import time
import os.path as osp

from dassl.utils.tools import mkdir_if_missing

# modify print function which will flush instantly
print = functools.partial(print, flush=True)


class Logger:
    def __init__(self, fpath=None, write_to_console=False):
        self.console = sys.stdout if write_to_console else None
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.console is not None:
            self.console.write(msg)

        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        if self.console is not None:
            self.console.flush()

        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        if self.console is not None:
            self.console.close()
            
        if self.file is not None:
            self.file.close()


def setup_logger(output=None, write_to_console=False):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath, write_to_console)
    return fpath
