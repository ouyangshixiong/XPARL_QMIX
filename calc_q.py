from qmixer_model import QMixerModel
from rnn_model import RNNModel
from qmix import QMIX
import parl

@parl.remote_class(wait=False)
class Calc(object):
    def __init__(self, config):
        self.agent_model = RNNModel(config['obs_shape'], config['n_actions'],
                           config['rnn_hidden_dim'])
        self.qmixer_model = QMixerModel(
                    config['n_agents'], config['state_shape'], config['mixing_embed_dim'],
                    config['hypernet_layers'], config['hypernet_embed_dim'])

        self.algorithm = QMIX(self.agent_model, self.qmixer_model, config['double_q'],
                    config['gamma'], config['lr'], config['clip_grad_norm'])

    def localQ(self, index, s_batch, obs_batch):
        #local_qs, target_local_qs = self.algorithm.localQ(s_batch, obs_batch)
        local_qs = []
        target_local_qs = []
        return index, local_qs, target_local_qs  