from typing import Iterator, List, Optional, NamedTuple, Dict, Any
from collections import deque
import dataclasses

import reverb
import numpy as np
import dm_env
from dm_env import specs
from acme import adders, specs, types
from acme.adders import reverb as adders_reverb
from acme.datasets import reverb as datasets


#################
### From Acme ###
#################

class Batch(NamedTuple):
    observation: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    discount: np.ndarray
    next_observation: np.ndarray
    extras: Dict

@dataclasses.dataclass
class ReverbReplay:
    server: reverb.Server
    adder: adders.Adder
    data_iterator: Iterator[reverb.ReplaySample]
    client: Optional[reverb.Client] = None

def make_replay_tables(
    environment_spec: specs.EnvironmentSpec,
    replay_table_name: str = 'replay buffer',
    max_replay_size: int = 2_000_000,
    min_replay_size: int = 100,
    extras_spec: types.NestedSpec = ()
) -> List[reverb.Table]:
    """Creates reverb tables for the algorithm."""
    return [reverb.Table(
        name=replay_table_name,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        max_size=max_replay_size,
        rate_limiter=reverb.rate_limiters.MinSize(min_replay_size),
        signature=adders_reverb.NStepTransitionAdder.signature(
            environment_spec, extras_spec))]

def make_dataset_iterator(
    replay_client: reverb.Client, batch_size: int,
    prefetch_size: int = 4, replay_table_name: str = 'replay buffer',
    ) -> Iterator[reverb.ReplaySample]:
    """Creates a dataset iterator to use for learning."""
    dataset = datasets.make_reverb_dataset(
        table=replay_table_name,
        server_address=replay_client.server_address,
        batch_size=batch_size,
        prefetch_size=prefetch_size)
    return dataset.as_numpy_iterator()

def make_adder(
    replay_client: reverb.Client,
    n_step: int, discount: float,
    replay_table_name: str = 'replay buffer',) -> adders.Adder:
    """Creates an adder which handles observations."""
    return adders_reverb.NStepTransitionAdder(
        priority_fns={replay_table_name: None},
        client=replay_client,
        n_step=n_step,
        discount=discount
    )

def get_reverb_replay_components(
    environment_spec: specs.EnvironmentSpec,
    n_step: int, discount: float, batch_size: int,
    max_replay_size: int = 2_000_000,
    min_replay_size: int = 100,
    replay_table_name: str = 'replay buffer',
    extras_spec: Optional[types.NestedSpec] = ()
    ) -> ReverbReplay:
    replay_table = make_replay_tables(environment_spec,
        replay_table_name, max_replay_size,
        min_replay_size=min_replay_size, extras_spec=extras_spec)
    server = reverb.Server(replay_table, port=None)
    address = f'localhost:{server.port}'
    client = reverb.Client(address)
    adder = make_adder(client, n_step, discount, replay_table_name)
    data_iterator = make_dataset_iterator(
        client, batch_size, replay_table_name=replay_table_name)
    return ReverbReplay(
        server, adder, data_iterator, client
    )


class IntraEpisodicBuffer:
    def __init__(self, maxlen: int = 1001, full_method: str = 'episodic') -> None:
        self.timesteps = deque(maxlen=maxlen)
        self.extras = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self.full_method = full_method
        self._last_timestep = None

    def add(self, timestep: dm_env.TimeStep, extra: Dict[str, Any]):
        self.timesteps.append(timestep)
        self.extras.append(extra)
        self._last_timestep = timestep

    def reset(self):
        self.timesteps = deque(maxlen=self._maxlen)
        self.extras = deque(maxlen=self._maxlen)
        self._last_timestep = None

    def __len__(self) -> int:
        return len(self.timesteps)

    def is_full(self):
        if self.full_method == 'episodic':
            # buffer is not full when just initialized/resetted
            if self._last_timestep is None:
                return False
            return self._last_timestep.last()
        if self.full_method == 'step':
            return len(self.timesteps) == self._maxlen
        raise NotImplementedError
