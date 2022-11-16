from typing import NamedTuple, OrderedDict, List, Tuple, Dict, Union
import datetime
import io
import random
import traceback
from collections import defaultdict
import pathlib
import functools

import numpy as np
import dm_env
import torch
from torch.utils.data import IterableDataset


class Batch(NamedTuple):
    observation: Union[np.ndarray, List]
    action: Union[np.ndarray, List]
    reward: Union[np.ndarray, List]
    discount: Union[np.ndarray, List]
    next_observation: Union[np.ndarray, List]
    extras: Dict  # List


def compute_episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read())


def _preload(replay_dir: pathlib.Path) -> Tuple[int, int]:
    """
    returns the number of episode and transitions in the replay_dir,
    it assumes that each episode's name has the format {}_{}_{episode_len}.npz
    """
    n_episodes, n_transitions = 0, 0
    for file in replay_dir.glob('*.npz'):
        _, _, episode_len = file.stem.split('_')
        n_episodes += 1
        n_transitions += int(episode_len)

    return n_episodes, n_transitions


class ReplayBufferStorage:
    def __init__(self,
                 data_specs, #: Tuple[specs, ...],
                 meta_specs, #: Tuple[specs, ...],
                 replay_dir: pathlib.Path = pathlib.Path.cwd() / 'buffer'
                 ):
        """
        data_specs: (obs, action , reward, discount)
        meta_specs: any extra e.g. (skill, mode)
        """

        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._n_episodes, self._n_transitions = _preload(replay_dir)

    def __len__(self):
        return self._n_transitions

    def add(self,
            time_step: dm_env.TimeStep,
            meta: OrderedDict
            ):
        self._add_meta(meta=meta)
        self._add_time_step(time_step=time_step)
        if time_step.last():
            self._store_episode()

    def _add_meta(self, meta: OrderedDict):
        for spec in self._meta_specs:
            value = meta[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            self._current_episode[spec.name].append(value)
        # for key, value in meta.items():
        #     self._current_episode[key].append(value)

    def _add_time_step(self, time_step: dm_env.TimeStep):
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                # convert it to a numpy array as shape given by the data specs (reward & discount)
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

    def _store_episode(self):
        episode = dict()

        # datas to save as numpy array
        for spec in self._data_specs:
            value = self._current_episode[spec.name]
            episode[spec.name] = np.array(value, spec.dtype)

        # metas to save as numpy array
        for spec in self._meta_specs:
            value = self._current_episode[spec.name]
            episode[spec.name] = np.array(value, spec.dtype)

        # reset current episode content
        self._current_episode = defaultdict(list)

        # save episode
        eps_idx = self._n_episodes
        eps_len = compute_episode_len(episode)
        self._n_episodes += 1
        self._n_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz'
        save_episode(episode, self._replay_dir / eps_fn)


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBuffer(IterableDataset):

    def __init__(self,
                 storage: ReplayBufferStorage,
                 max_size: int,
                 num_workers: int,
                 nstep: int,
                 discount: int,
                 fetch_every: int,
                 save_snapshot: bool
    ):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def __len__(self):
        return len(self._storage)

    def _sample_episode(self):
        """ Sample a single episode """
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        """
        load an episode in memory with dict self._episodes
        and self._episode_fns contains the sorted keys
        and deletes the file
        """
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = compute_episode_len(episode)
        # remove old episodes if max size is reached
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= compute_episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        # store the episode
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        # delete episode if save_snapshot false
        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        """
        Fetch all episodes, divided between workers
        """
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0

        # last created to first created
        eps_fns = sorted(self._storage._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        # load all episodes
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            # each worker load an episode
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            # stop if fail to load episode
            if not self._store_episode(eps_fn):
                break

    def _sample(self
):  # -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ...]:
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition so only starts above 1 and below max-nstep
        # idx to take inside the episode
        idx = np.random.randint(0, compute_episode_len(episode) - self._nstep + 1) + 1
        meta = dict()
        # meta = []
        for spec in self._storage._meta_specs:
            meta[spec.name] = episode[spec.name][idx - 1]
            # meta.append(episode[spec.name][idx - 1])
        obs = episode['observation'][idx - 1] # account for first dummy transition
        action = episode['action'][idx] # on first dummy transition action is set to 0
        next_obs = episode['observation'][idx + self._nstep - 1]# account for first dummy transition
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        # noinspection PyRedundantParentheses
        data = dict(
            observation=obs,
            action=action,
            reward=reward,
            discount=discount,
            next_observation=next_obs,
        )
        data.update(meta)
        return data
        # return (obs, action, reward, discount, next_obs, *meta)

    def __iter__(self):
        while True:
            yield self._sample()


class RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def numpy_collate(batch: List[Dict], meta_specs):
    res = defaultdict(list)
    for b in batch:
        for k, v in b.items():
            res[k].append(v)
    extras = dict()
    for spec in meta_specs:
        extras[spec.name] = np.stack(res[spec.name])
    return Batch(
        observation=np.stack(res['observation']),
        action=np.stack(res['action']),
        reward=np.stack(res['reward']),
        discount=np.stack(res['discount']),
        next_observation=np.stack(res['next_observation']),
        extras=extras
    )

def numpy_collate_mode(batch: List[Dict], meta_specs):
    res_mode0 = defaultdict(list)
    res_mode1 = defaultdict(list)
    for b in batch:

        if b['mode'] == 0:
            res_mode0['skill'].append(b['skill'])
            res_mode0['observation'].append(b['observation'])
            res_mode0['next_observation'].append(b['next_observation'])
            res_mode0['action'].append(b['action'])
            res_mode0['reward'].append(b['reward'])
            res_mode0['discount'].append(b['discount'])
        elif b['mode'] == 1:
            res_mode1['skill'].append(b['skill'])
            res_mode1['observation'].append(b['observation'])
            res_mode1['next_observation'].append(b['next_observation'])
            res_mode1['action'].append(b['action'])
            res_mode1['reward'].append(b['reward'])
            # res_mode1['discount'].append(b['discount'] * 0.25)
            res_mode1['discount'].append(b['discount'])

    extras = dict()
    # for spec in meta_specs:
    extras['skill'] = [np.stack(res_mode0['skill']), np.stack(res_mode1['skill'])]
    # extras['skill'] = [] #[np.stack(res_mode0[spec.name]),  np.stack(res_mode1[spec.name])]
    # if len(res_mode0['skill']):
    #     extras['skill'].append(np.stack(res_mode0['skill']))
    # if len(res_mode1['skill']):
    #     extras['skill'].append(np.stack(res_mode1['skill']))

    return Batch(
        observation=[np.stack(res_mode0['observation']), np.stack(res_mode1['observation'])],
        action=[np.stack(res_mode0['action']), np.stack(res_mode1['action'])],
        reward=[np.stack(res_mode0['reward']), np.stack(res_mode1['reward'])],
        discount=[np.stack(res_mode0['discount']), np.stack(res_mode1['discount'])],
        next_observation=[np.stack(res_mode0['next_observation']), np.stack(res_mode1['next_observation'])],
        extras=extras
    )


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(storage,
                       max_size,
                       batch_size,
                       num_workers,
                       nstep,
                       discount,
                       meta_specs,
                       save_snapshot: bool = False):


    if 'mode' in [spec.name for spec in meta_specs]:
        # collate_fct = functools.partial(numpy_collate, meta_specs=meta_specs)
        collate_fct = functools.partial(numpy_collate_mode, meta_specs=meta_specs)
    else:
        collate_fct = functools.partial(numpy_collate, meta_specs=meta_specs)

    max_size_per_worker = max_size // max(1, num_workers)
    iterable = ReplayBuffer(storage,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True,
                                         worker_init_fn=_worker_init_fn,
                                         collate_fn=collate_fct
                                         )
    return loader
