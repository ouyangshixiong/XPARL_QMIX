import os
import time
from smac.env import StarCraft2Env
from env_wrapper import SC2EnvWrapper
from replay_buffer import EpisodeExperience, EpisodeReplayBuffer
from qmixer_model import QMixerModel
from rnn_model import RNNModel
from qmix import QMIX
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
        self.env = SC2EnvWrapper(env)
        config['episode_limit'] = self.env.episode_limit
        config['obs_shape'] = self.env.obs_shape
        config['state_shape'] = self.env.state_shape
        config['n_agents'] = self.env.n_agents
        config['n_actions'] = self.env.n_actions
        self.config = deepcopy(config)

        self.agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])
        self.qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])

        self.algorithm = QMIX(self.agent_model, self.qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])

        self.qmix_agent = QMixAgent(
                    self.algorithm, config['exploration_start'], config['min_exploration'],
                    config['exploration_decay'], config['update_target_interval'])
        #=== Learner ===
        #self.total_steps = 0
        self.central_steps = 0
        self.learn_steps = 0
        self.target_update_count = 0
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
        self.central_steps += 1
        # get the total train data of all the actors.
        sample_data_object_ids = [
            remote_actor.sample() for remote_actor in self.remote_actors
        ]
        sample_datas = [
            future_object.get() for future_object in sample_data_object_ids
        ]
        mean_loss = []
        mean_td_error = []
        for sample_data in sample_datas:
            for data in sample_data:
                #if 'steps' == data:
                    #for steps in sample_data[data]:
                        #self.total_steps += steps
                #elif 'episode_experience' == data:
                if 'episode_experience' == data:
                    for episode_experience in sample_data[data]:
                        self.rpm.add(episode_experience)
            if self.rpm.count > self.config['memory_warmup_size']:
                # exec slow code in remote actor, calc local_qs, target_local_qs
                futureObjs = []
                rpm_sample = defaultdict(list)
                for index in range(self.config['sample_batch_episode']):
                    self.learn_steps += 1
                    s_batch, a_batch, r_batch, t_batch, obs_batch, available_actions_batch,\
                            filled_batch = self.rpm.sample_batch(self.config['batch_size'])
                    rpm_sample[index] = [s_batch, a_batch, r_batch, t_batch, available_actions_batch, filled_batch]
                    #local_qs, target_local_qs
                    print(type(index))
                    print(type(s_batch))
                    print(type(obs_batch.tolist()))
                    futureObj = self.remote_actors[index].localQ(index, s_batch, obs_batch.tolist())
                    futureObjs.append(futureObj)

                for futureObj in futureObjs:
                    index, local_qs, target_local_qs = futureObj.get()
                    s_batch, a_batch, r_batch, t_batch, available_actions_batch, filled_batch = rpm_sample[index]
                    #learn
                    loss, td_error = self.qmix_agent.learn(s_batch, a_batch, r_batch, t_batch,
                                            available_actions_batch, filled_batch,
                                            local_qs, target_local_qs)
                    # update remote networks
                    if self.learn_steps % self.config['update_target_interval'] == 0:
                        update_target_q = True
                        agent_target = self.algorithm.target_agent_model.get_weights()
                        qmix_target = self.algorithm.target_qmixer_model.get_weights()
                        self.target_update_count += 1
                    else:
                        update_target_q = False

                    for remote_actor in self.remote_actors:
                        agent_network_params = self.agent_model.get_weights()
                        qmix_network_params = self.qmixer_model.get_weights()
                        remote_actor.update_network(agent_network_params, qmix_network_params)
                        if update_target_q:
                            remote_actor.update_target_network(agent_target, qmix_target)
                        
                    mean_loss.append(loss)
                    mean_td_error.append(td_error)

        mean_loss = np.mean(mean_loss) if mean_loss else None
        mean_td_error = np.mean(mean_td_error) if mean_td_error else None
        return mean_loss, mean_td_error


    def should_stop(self):
        #return self.total_steps >= self.config['training_steps']
        return self.central_steps >= self.config['training_steps']

    def run_evaluate_episode(self):
        self.qmix_agent.reset_agent()
        episode_reward = 0.0
        episode_step = 0
        terminated = False
        state, obs = self.env.reset()

        while not terminated:
            available_actions = self.env.get_available_actions()
            actions = self.qmix_agent.predict(obs, available_actions)
            state, obs, reward, terminated = self.env.step(actions)
            episode_step += 1
            episode_reward += reward

        is_win = self.env.win_counted
        return episode_reward, episode_step, is_win

    def save(self):
        self.env.save()


if __name__ == '__main__':
    from qmix_config import QMixConfig as config
    learner = Learner(config)
    # warm up
    while learner.rpm.count < config['memory_warmup_size']:
        learner.step()

    while not learner.should_stop():
        loss, td_error = learner.step()
        summary.add_scalar('train_loss', loss, learner.central_steps)
        summary.add_scalar('train_td_error:', td_error, learner.central_steps)

        if learner.central_steps % config['test_steps'] == 0:
            eval_reward_buffer = []
            eval_steps_buffer = []
            eval_is_win_buffer = []
            for _ in range(3):
                eval_reward, eval_step, eval_is_win = learner.run_evaluate_episode()
                eval_reward_buffer.append(eval_reward)
                eval_steps_buffer.append(eval_step)
                eval_is_win_buffer.append(eval_is_win)
            summary.add_scalar('eval_reward', np.mean(eval_reward_buffer),
                            learner.central_steps)
            summary.add_scalar('eval_steps', np.mean(eval_steps_buffer),
                            learner.central_steps)
            mean_win_rate = np.mean(eval_is_win_buffer)
            summary.add_scalar('eval_win_rate', mean_win_rate,
                            learner.central_steps)
            summary.add_scalar('target_update_count',
                            learner.target_update_count, learner.central_steps)
            #if mean_win_rate == 1:
                #print("save replay!")
                #learner.save()
                    

        
