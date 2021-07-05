from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from parl.algorithms import QMIX
from qmix_agent import QMixAgent
from replay_buffer import EpisodeExperience
import numpy as np
import parl
from collections import defaultdict

@parl.remote_class(wait=False)
class Actor(object):
    def __init__(self, config):
        self.config = config

        env = StarCraft2Env(
                map_name=config['scenario'], difficulty=config['difficulty'])
        self.env = SC2EnvWrapper(env)

        self.state, self.obs = self.env.reset()

        self.config['episode_limit'] = self.env.episode_limit
        self.config['obs_shape'] = self.env.obs_shape
        self.config['state_shape'] = self.env.state_shape
        self.config['n_agents'] = self.env.n_agents
        self.config['n_actions'] = self.env.n_actions

        agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])
        # 不需要的                   
        qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])

        algorithm = QMIX(agent_model, qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])

        self.qmix_agent = QMixAgent(
                    algorithm, config['exploration_start'], config['min_exploration'],
                    config['exploration_decay'], config['update_target_interval'])

    def sample(self):
        sample_data = defaultdict(list)
        for i in range(self.config['sample_batch_steps']):
            available_actions = self.env.get_available_actions()
            actions = self.qmix_agent.sample(self.obs, available_actions)
            next_state, next_obs, reward, terminated  = self.env.step(actions)
            sample_data['state'].append(self.state)
            sample_data['obs'].append(self.obs)
            sample_data['actions'].append(actions)
            sample_data['rewards'].append(reward)
            sample_data['terminated'].append(terminated)
            sample_data['available_actions'].append(available_actions)
            self.state = next_state
            self.obs = next_obs

        # size of sample_data: 1 * sample_batch_steps
        for key in sample_data:
            sample_data[key] = np.stack(sample_data[key])

        return sample_data

