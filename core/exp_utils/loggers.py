import random
import timeit
import time
import contextlib
import logging
from collections import defaultdict

import haiku as hk
import csv
import torch
import numpy as np
import jax.numpy as jnp
import wandb
from pathlib import Path

#TODO remove those
def log_params_to_wandb(params: hk.Params, step: int):
    if params:
        for module in sorted(params):
            if 'w' in params[module]:
                wandb.log({
                    f'{module}/w': wandb.Histogram(params[module]['w'])
                }, step=step)
            if 'b' in params[module]:
                wandb.log({
                    f'{module}/b': wandb.Histogram(params[module]['b'])
                }, step=step)

class LogParamsEvery:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, params: hk.Params, step):
        if self._every is None:
            pass
        every = self._every // self._action_repeat
        if step % every == 0:
            log_params_to_wandb(params, step)
        pass

class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time

@contextlib.contextmanager
def time_activity(activity_name: str):
    logger = logging.getLogger(__name__)
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    logger.info('[Timing] %s finished (Took %.2fs).', activity_name, duration)

class AverageMeter:
    def __init__(self):
        self._sum = 0.
        self._count = 0
        self.fmt = "{value:.4f}"

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    @property
    def value(self):
        return self._sum / max(1, self._count)

    def __str__(self):
        return self.fmt.format(
            value=self.value
        )

def dict_to_header(data: dict, header=None):
    if header is not None:
        header = [header]
    else:
        header = []
    delimiter = '\t'
    for name, value in data.items():
        if type(value) == float:
            header.append(
                '{}: {:.4f}'.format(name, value)
            )
        elif type(value) == np.ndarray: # reward is a np.ndarray of shape ()
            header.append(
                '{}: {:.4f}'.format(name, value)
            )
        else:
            header.append(
                '{}: {}'.format(name, value)
            )
    return delimiter.join(header)

class MetricLogger:
    def __init__(self,
                 csv_file_name: Path,
                 use_wandb: bool,
                 delimiter= "\t"
    ):
        self.logger = logging.getLogger(__name__)
        self._meters = defaultdict(AverageMeter) # factory
        self._csv_writer = None
        self._csv_file = None
        self._csv_file_name = csv_file_name
        self.delimiter = delimiter
        self.use_wandb = use_wandb

    def update_metrics(self,**kwargs):
        """Log the average of variables that are logged per episode"""
        for k, v in kwargs.items():
            if isinstance(v, jnp.DeviceArray):
                v = v.item()
            assert isinstance(v, (float, int))
            self._meters[k].update(v)

    def log_and_dump_metrics_to_wandb(self, step: int, header=''):
        """log and dump to wandb metrics"""
        if type(header) == dict:
            header = dict_to_header(data=header)
        self.logger.info(self._log_meters(header=header))
        if self.use_wandb:
            for name, meter in self._meters.items():
                wandb.log({name: np.mean(meter.value).item()}, step=step)
        self._clean_meters()

    def _clean_meters(self):
        self._meters.clear()

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']: # assume episode exist in header of existing file
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def dump_dict_to_csv(self, data: dict):
        """dump to wandb and csv the dict"""
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists(): # if file already exists remove entries
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=sorted(data.keys()),
                restval=0.0
            )
            if should_write_header:
                self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def dump_dict_to_wandb(self, step: int, data: dict):
        for name, value in data.items():
            if self.use_wandb:
                wandb.log({name: np.mean(value).item()}, step=step)

    def log_dict(self, header, data):
        self.logger.info(dict_to_header(data=data, header=header))

    def _log_meters(self, header: str):
        loss_str = [header]
        for name, meter in self._meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
