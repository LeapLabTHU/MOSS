from abc import ABC, abstractmethod



class IntrinsicReward(ABC):

    @abstractmethod
    def init_params(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def compute_reward(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update_batch(self, *args, **kwargs):
        raise NotImplementedError
