import os
import time
from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from replay_buffer import EpisodeExperience, EpisodeReplayBuffer
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from parl.algorithms import QMIX
from qmix_agent import QMixAgent
import parl
from parl.utils import logger
from parl.utils import summary
import numpy as np
from copy import deepcopy
from collections import defaultdict

from actor import Actor

class Learner(object):
    def __init__(self, config):
        #=== Create Agent ===
        env = StarCraft2Env(
            map_name=config['scenario'], difficulty=config['difficulty'])
        env = SC2EnvWrapper(env)
        config['episode_limit'] = env.episode_limit
        config['obs_shape'] = env.obs_shape
        config['state_shape'] = env.state_shape
        config['n_agents'] = env.n_agents
        config['n_actions'] = env.n_actions
        self.config = deepcopy(config)

        agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])
        qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])

        algorithm = QMIX(agent_model, qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])

        self.qmix_agent = QMixAgent(
                    algorithm, config['exploration_start'], config['min_exploration'],
                    config['exploration_decay'], config['update_target_interval'])
        #=== Learner ===
        self.total_steps = 0
        self.rpm = EpisodeReplayBuffer(config['replay_buffer_size'])

        #=== Remote Actor ===
        self.create_actors()

    def create_actors(self):
        parl.connect(self.config['master_address'])
        self.remote_actors = [
            Actor(self.config) for _ in range(self.config['actor_num'])
        ]
        logger.info('Creating {} remote actors to connect.'.format(
            self.config['actor_num']))
        self.start_time = time.time()

    def step(self):
        # get the total train data of all the actors.
        sample_data_object_ids = [
            remote_actor.sample() for remote_actor in self.remote_actors
        ]
        sample_datas = [
            future_object.get() for future_object in sample_data_object_ids
        ]
        for sample_data in sample_datas:
            for data in sample_data:
                if 'steps' == data:
                    for steps in sample_data[data]:
                        self.total_steps += steps
                elif 'episode_experience' == data:
                    for episode_experience in sample_data[data]:
                        self.rpm.add(episode_experience)

        mean_loss = []
        mean_td_error = []
        if self.rpm.count > config['memory_warmup_size']:
            for _ in range(2):
                s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch,\
                        filled_batch = self.rpm.sample_batch(config['batch_size'])
                loss, td_error = self.qmix_agent.learn(s_batch, a_batch, r_batch, t_batch,
                                            obs_batch, available_actions_batch,
                                            filled_batch)
                mean_loss.append(loss)
                mean_td_error.append(td_error)

        mean_loss = np.mean(mean_loss) if mean_loss else None
        mean_td_error = np.mean(mean_td_error) if mean_td_error else None
        return mean_loss, mean_td_error


    def should_stop(self):
        return self.total_steps >= self.config['training_steps']


if __name__ == '__main__':
    from qmix_config import QMixConfig as config
    learner = Learner(config)
    # warm up
    while learner.rpm.count < config['memory_warmup_size']:
        learner.step()

    while not learner.should_stop():
        start = time.time()
        while time.time() - start < config['log_metrics_interval_s']:
            loss, td_error = learner.step()
            summary.add_scalar('train_loss', loss, learner.total_steps)
            summary.add_scalar('train_td_error:', td_error, learner.total_steps)
        