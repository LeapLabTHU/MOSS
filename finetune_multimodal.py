import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import random
import logging
from typing import NamedTuple

import wandb
import jax
import torch
import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig
import pathlib
from dm_env import specs
import jax.numpy as jnp
from tqdm import trange

from core import envs
from core import agents
from core import exp_utils
import helpers

class RunningStatistics(NamedTuple):
    mode_0_running_mean: jnp.ndarray
    mode_0_running_std: jnp.ndarray
    mode_0_running_num: float


class FinetuneLoop:
    def __init__(self, cfg):
        self.logger = logging.getLogger(__name__)
        self.logger.info(OmegaConf.to_yaml(cfg))
        self.init_rng = jax.random.PRNGKey(cfg.seed)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.work_dir = pathlib.Path.cwd()
        self.use_wandb = cfg.use_wandb
        if cfg.use_wandb:
            WANDB_NOTES = cfg.wandb_note
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(project=cfg.wandb_project_name,
                       name='finetune_'+cfg.benchmark.task+WANDB_NOTES+str(cfg.seed),
                       config=OmegaConf.to_container(cfg, resolve=True))

        # init env
        self.train_env = envs.make_env(cfg.agent.action_type, cfg.benchmark, seed=cfg.seed)
        self.eval_env = envs.make_env(cfg.agent.action_type, cfg.benchmark, seed=cfg.seed)
        self.action_repeat = cfg.benchmark.action_repeat

        # init agent
        self.agent = agents.make_agent(
            cfg.benchmark.obs_type,
            self.train_env.action_shape,
            cfg.agent
        )
        data_specs = (
        self.train_env.observation_spec(),
        self.train_env.action_spec(),
        specs.Array((1,), np.float32, 'reward'),
        specs.Array((1,), np.float32, 'discount'),
        )
        self.agent.init_replay_buffer(
        replay_buffer_cfg=cfg.agent.replay_buffer_cfg,
        replay_dir=self.work_dir / 'buffer',
        environment_spec=data_specs,
        )
        self.update_agent_every = cfg.update_every_steps

        self.checkpointer = exp_utils.Checkpointer(save_dir=cfg.save_dir)
        checkpoint = self.checkpointer.load_checkpoint(path=cfg.checkpoint)
        self.state = checkpoint.get('state')

        self.intrinsic_reward = hydra.utils.instantiate(cfg.intrinsic)

        # init exp_utils
        self.train_video_recorder = exp_utils.TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None  # if state based no pixels to save
        )
        self.video_recorder = exp_utils.VideoRecorder(
            self.work_dir if cfg.save_video else None,
            use_wandb=cfg.use_wandb
        )

        # init loop
        eval_until_episode = helpers.Until(cfg.num_eval_episodes, cfg.benchmark.action_repeat)
        seed_until_step = helpers.Until(cfg.num_seed_frames, cfg.benchmark.action_repeat)
        eval_every_step = helpers.Every(cfg.eval_every_frames, cfg.benchmark.action_repeat)
        train_until_step = helpers.Until(cfg.num_finetune_frames, cfg.benchmark.action_repeat)
        self.loops_length = helpers.LoopsLength(
            eval_until_episode=eval_until_episode,
            train_until_step=train_until_step,
            seed_until_step=seed_until_step,
            eval_every_step=eval_every_step,
        )
        self.global_loop_var = helpers.LoopVar(
                        global_step=0,
                        global_episode=0,
                        episode_step=0,
                        episode_reward=0.,
                        total_reward=0.,
                        pointer=0)
        self.skill_tracker = helpers.init_skilltracker(
            search_steps=self.loops_length.seed_until_step._until,
            change_interval=cfg.agent.skills_cfg.update_skill_every,
            low=0.
        )

    @property
    def global_frame(self):
        return self.global_loop_var.global_step * self.action_repeat

    def _exploration_loop(self, rng):
        time_step = self.train_env.reset()
        meta_rng, rng = jax.random.split(key=rng, num=2)
        self.skill_tracker = helpers.skilltracker_update_on(self.skill_tracker)
        meta, self.skill_tracker = self.agent.init_meta(key=meta_rng,
                                    time_step=time_step,
                                    step=self.global_loop_var.global_step,
                                    skill_tracker=self.skill_tracker)

        self.agent.store_timestep(time_step=time_step, meta=meta)
        while self.loops_length.seed_until_step(self.global_loop_var.global_step):
            if time_step.last():
                self.global_loop_var = helpers.increment_episode(self.global_loop_var)
                time_step = self.train_env.reset()
                meta_rng, rng = jax.random.split(key=rng, num=2)
                meta, self.skill_tracker = self.agent.init_meta(key=meta_rng, time_step=time_step,
                                                                step=self.global_loop_var.global_step,
                                                                skill_tracker=self.skill_tracker
                                                                )
                self.agent.store_timestep(time_step=time_step, meta=meta)
                self.global_loop_var = helpers.reset_episode(self.global_loop_var)

            meta_rng, action_rng, rng = tuple(jax.random.split(key=rng, num=3))
            meta, self.skill_tracker = self.agent.update_meta(key=meta_rng,
                                                              meta=meta,
                                                              step=self.global_loop_var.global_step,
                                                              time_step=time_step,
                                                              skill_tracker=self.skill_tracker)
            action = self.agent.select_action(
                state=self.state,
                obs=time_step.observation,
                meta=meta,
                step=self.global_loop_var.global_step,
                key=action_rng,
                greedy=True
            )
            action = np.array(action)

            # take env step
            time_step = self.train_env.step(action)
            self.agent.store_timestep(time_step=time_step, meta=meta)
            # increment loop_vars
            self.global_loop_var = helpers.increment_step(self.global_loop_var, reward=time_step.reward)
            self.skill_tracker = helpers.update_skilltracker(self.skill_tracker, time_step.reward)

        self.skill_tracker = helpers.skilltracker_update_off(self.skill_tracker)
        return time_step, meta

    def train_loop(self):

        metric_logger = exp_utils.MetricLogger(csv_file_name=self.work_dir / 'train.csv', use_wandb=self.use_wandb)
        timer = exp_utils.Timer()
        time_step = self.train_env.reset()

        self.state = self.agent.init_params(
            init_key=self.init_rng,
            dummy_obs=time_step.observation,
            summarize=True,
            checkpoint_state=self.state
        )

        step_rng, eval_rng, rng = jax.random.split(self.init_rng, num=3)
        self.logger.info("Exploration loop")
        time_step, meta = self._exploration_loop(step_rng)
        self.evaluate(eval_rng)
        self.logger.info("Best skill vector: {}".format(self.skill_tracker.best_skill))
        self.logger.info("Starting training at episode: {}, step: {}".format(self.global_loop_var.global_episode,
                                                                             self.global_loop_var.global_step))

        metrics = None
        while self.loops_length.train_until_step(self.global_loop_var.global_step):
            if time_step.last():
                self.global_loop_var = helpers.increment_episode(self.global_loop_var)

                # log metrics
                if metrics is not None:
                    elapsed_time, total_time = timer.reset()
                    episode_frame = self.global_loop_var.episode_step * self.action_repeat
                    data = helpers.CsvData(
                        step=self.global_loop_var.global_step,
                        episode=self.global_loop_var.global_episode,
                        episode_length=episode_frame,
                        episode_reward=self.global_loop_var.episode_reward,  # not a float type
                        total_time=total_time,
                        fps=episode_frame / elapsed_time
                    )
                    data = data._asdict()
                    metric_logger.dump_dict_to_csv(data=data)
                    metric_logger.dump_dict_to_wandb(step=self.global_loop_var.global_step, data=data)
                    data.update(buffer_size=len(self.agent))
                    metric_logger.log_and_dump_metrics_to_wandb(step=self.global_loop_var.global_step, header=data)

                # reset env
                time_step = self.train_env.reset()
                step_rng, rng = tuple(jax.random.split(rng, num=2))
                meta, self.skill_tracker = self.agent.init_meta(key=step_rng,
                                            time_step=time_step,
                                            step=self.global_loop_var.global_step,
                                            skill_tracker=self.skill_tracker
                                            )
                # no need to parse because not updating it duing finetune loop
                self.agent.store_timestep(time_step=time_step, meta=meta)
                # train_video_recorder.init(time_step.observation)
                self.global_loop_var = helpers.reset_episode(self.global_loop_var)

            if self.loops_length.eval_every_step(self.global_loop_var.global_step):
                eval_rng, rng = tuple(jax.random.split(rng, num=2))
                self.evaluate(eval_rng=eval_rng)

            # agent step
            meta_rng, step_rng, update_rng, rng = tuple(jax.random.split(rng, num=4))
            meta, self.skill_tracker = self.agent.update_meta(
                key=meta_rng,
                meta=meta,
                step=self.global_loop_var.global_step,
                time_step=time_step,
                skill_tracker=self.skill_tracker)
            action = self.agent.select_action(
                state=self.state,
                obs=time_step.observation,
                meta=meta,
                step=self.global_loop_var.global_step,
                key=step_rng,
                greedy=False
            )
            if self.global_loop_var.global_step % self.update_agent_every == 0:
                batch = self.agent.sample_timesteps()
                self.state, metrics = self.agent.update(
                    state=self.state,
                    key=update_rng,
                    step=self.global_loop_var.global_step,
                    batch=batch
                )
                metric_logger.update_metrics(**metrics)

            # step on env
            time_step = self.train_env.step(action)
            self.agent.store_timestep(time_step=time_step, meta=meta)
            self.global_loop_var = helpers.increment_step(self.global_loop_var, reward=time_step.reward)

        eval_rng, rng = jax.random.split(rng, num=2)
        self.evaluate(eval_rng=eval_rng)

    def evaluate(self, eval_rng):
        metric_logger = exp_utils.MetricLogger(csv_file_name=pathlib.Path.cwd() / 'eval.csv', use_wandb=self.use_wandb)
        timer = exp_utils.Timer()
        local_loop_var = helpers.LoopVar(
            global_step=0,
            global_episode=0,
            episode_step=0,
            episode_reward=0.,
            total_reward=0.,
            pointer=0,
        )
        while self.loops_length.eval_until_episode(local_loop_var.global_episode):
            step_rng, rng = jax.random.split(key=eval_rng, num=2)
            time_step = self.eval_env.reset()
            meta, self.skill_tracker = self.agent.init_meta(key=step_rng,
                                                            step=self.global_loop_var.global_step,
                                                            time_step=time_step,
                                                            skill_tracker=self.skill_tracker)
            self.video_recorder.init(self.eval_env, enabled=(local_loop_var.global_episode == 0))
            while not time_step.last():
                action = self.agent.select_action(
                    state=self.state,
                    obs=time_step.observation,
                    meta=meta,
                    step=self.global_loop_var.global_step,
                    key=step_rng,
                    greedy=True
                )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                local_loop_var = helpers.increment_step(local_loop_var, reward=time_step.reward)

            # episode += 1
            local_loop_var = helpers.increment_episode(local_loop_var)
            self.video_recorder.save(f'{self.global_loop_var.global_step * self.action_repeat}.mp4',
                                     step=self.global_loop_var.global_step)

        n_frame = local_loop_var.global_step * self.action_repeat
        total_time = timer.total_time()
        data = helpers.CsvData(
            episode_reward=local_loop_var.total_reward / local_loop_var.global_episode,
            episode_length=int(local_loop_var.global_step * self.action_repeat / local_loop_var.global_episode),
            episode=self.global_loop_var.global_episode,  # must name it episode otherwise the csv cannot clean it
            step=self.global_loop_var.global_step,
            total_time=total_time,
            fps=n_frame / total_time
        )
        data = data._asdict()
        metric_logger.dump_dict_to_csv(data=data)
        metric_logger.dump_dict_to_wandb(step=self.global_loop_var.global_step, data=data)
        metric_logger.log_dict(data=data, header="Evaluation results: ")
        return data


@hydra.main(config_path='conf/', config_name='config')
def main(cfg: DictConfig):
    trainer = FinetuneLoop(cfg)
    trainer.train_loop()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)  # dmc version
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # resolves tf/jax concurrent use conflict
    main()