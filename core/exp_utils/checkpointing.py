import logging
import os
from typing import Any, Mapping, Text
from functools import partial

import dill
import jax
import jax.numpy as jnp
from iopath.common.file_io import PathManager

# from core.misc.utils import broadcast

logger = logging.getLogger(__name__)
path_manager = PathManager()

def tag_last_checkpoint(save_dir: str,
                        last_filename_basename: str) -> None:
    """ save name of the last checkpoint in the file `last_checkpoint` """
    save_file = os.path.join(save_dir, "last_checkpoint")
    with path_manager.open(save_file, "w") as f:
        f.write(last_filename_basename)

def save_state(save_dir: str,
               name: str,
               state: Mapping[Text, jnp.ndarray],
               step: int,
               rng,
               **kwargs: Any) -> None:
    n_devices = jax.local_device_count()
    if jax.process_index() != 0: # only checkpoint the first worker
        return
    checkpoint_data = dict(
        # state=state,
        state= jax.tree_map(
            lambda x: jax.device_get(x[0]) if n_devices > 1 else jax.device_get(x), state),
        step=step,
        rng=rng
    )
    checkpoint_data.update(kwargs)
    basename = "{}.pth".format(name)
    save_file = os.path.join(save_dir, basename)
    assert os.path.basename(save_file) == basename, basename
    logger.info("Saving checkpoint to {}".format(save_file))
    with path_manager.open(save_file, "wb") as f:
        dill.dump(checkpoint_data, f)
    # tag it for auto resuming
    tag_last_checkpoint(
        save_dir=save_dir,
        last_filename_basename=basename,
    )

def has_last_checkpoint(save_dir:str) -> bool:
    save_dir = os.path.join(save_dir, "last_checkpoint")
    return path_manager.exists(save_dir)

def get_last_checkpoint(save_dir: str) -> str:
    save_file = os.path.join(save_dir, "last_checkpoint")
    try:
        with path_manager.open(save_file, "r") as f:
            last_saved = f.read().strip()
    except IOError:
        # if file doesn't exist, maybe because it has just been
        # deleted by a separate process
        return ""
    return os.path.join(save_dir, last_saved)

def resume_or_load(path: str, save_dir, *, resume: bool = False):
    if resume and has_last_checkpoint(save_dir):
        path = get_last_checkpoint(save_dir)
        return load_checkpoint(path)
    else:
        return load_checkpoint(path)

def load_checkpoint(path: str) -> Mapping[str, Any]:
    """
    :param path:
    :return: empty dict if checkpoint doesn't exist
    """
    if not path:
        logger.info("No checkpoint given.")
        return dict()

    if not os.path.isfile(path):
        path = path_manager.get_local_path(path)
        assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

    with path_manager.open(path, 'rb') as checkpoint_file:
        checkpoint = dill.load(checkpoint_file)
        logger.info('Loading checkpoint from %s', checkpoint_file)

    return checkpoint

class Checkpointer:
    def __init__(self,
                 save_dir: str = "checkpoints",
                 ):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_state = partial(save_state, save_dir=save_dir)
        self.load_checkpoint = load_checkpoint
        self.resume_or_load = partial(resume_or_load, save_dir=save_dir)
