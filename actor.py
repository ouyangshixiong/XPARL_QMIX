from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from qmix import QMIX
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

        #self.state, self.obs = self.env.reset()

        self.config['episode_limit'] = self.env.episode_limit
        self.config['obs_shape'] = self.env.obs_shape
        self.config['state_shape'] = self.env.state_shape
        self.config['n_agents'] = self.env.n_agents
        self.config['n_actions'] = self.env.n_actions

        self.agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])                  
        self.qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])

        algorithm = QMIX(self.agent_model, self.qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])

        self.qmix_agent = QMixAgent(
                    algorithm, config['exploration_start'], config['min_exploration'],
                    config['exploration_decay'], config['update_target_interval'])

    def sample(self):
        sample_data = defaultdict(list)
        for i in range(self.config['sample_batch_episode']):
            self.qmix_agent.reset_agent()
            episode_reward = 0.0
            episode_step = 0
            terminated = False
            state, obs = self.env.reset()
            episode_experience = EpisodeExperience(self.config['episode_limit'])

            while not terminated:
                available_actions = self.env.get_available_actions()
                actions = self.qmix_agent.sample(obs, available_actions)
                next_state, next_obs, reward, terminated = self.env.step(actions)
                episode_reward += reward
                episode_step += 1
                episode_experience.add(state, actions, [reward], [terminated], obs,
                                    available_actions, [0])
                state = next_state
                obs = next_obs

            sample_data['steps'].extend([episode_experience.count])
            # fullfill
            # to reduce computing resouce in learnning thread, put these lines here
            state_zero = np.zeros_like(state, dtype=state.dtype)
            actions_zero = np.zeros_like(actions, dtype=actions.dtype)
            obs_zero = np.zeros_like(obs, dtype=obs.dtype)
            available_actions_zero = np.zeros_like(
                available_actions, dtype=available_actions.dtype)
            reward_zero = 0
            terminated_zero = True
            for _ in range(episode_experience.count, self.config['episode_limit']):
                episode_experience.add(state_zero, actions_zero, [reward_zero],
                                    [terminated_zero], obs_zero,
                                    available_actions_zero, [1])
            sample_data['episode_experience'].extend([episode_experience])
        return sample_data
    def set_weights(self, agent_params, qmix_params):
        self.agent_model.set_weights(agent_params)
        self.qmixer_model.set_weights(qmix_params)

