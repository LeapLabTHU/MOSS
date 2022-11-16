from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def init_params(self,
                    init_key,
                    dummy_obs,
                    summarize = True
    ):
        raise NotImplementedError

    @abstractmethod
    def select_action(self, *args, **kwargs):
        """act function"""
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_meta_specs(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def init_meta(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_meta(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def init_replay_buffer(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def store_timestep(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def sample_timesteps(self, *args, **kwargs):
        raise NotImplementedError
