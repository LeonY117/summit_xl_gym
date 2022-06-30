from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np


class customNetBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            network_builder.BaseNetwork.__init__(self)
            self.load(params)
            return

        def load(self, params):
            self.separate = params.get('separate', False)
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            # self.is_d2rl = params['mlp'].get('d2rl', False)
            # self.norm_only_first_layer = params['mlp'].get(
            #     'norm_only_first_layer', False)
            # self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            # self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            # self.central_value = params.get('central_value', False)
            # self.joint_obs_actions_config = params.get(
            #     'joint_obs_actions', None)

            if self.has_space:
                self.is_multi_discrete = 'multi_discrete' in params['space']
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous' in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
                elif self.is_multi_discrete:
                    self.space_config = params['space']['multi_discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            # if self.has_rnn:
            #     self.rnn_units = params['rnn']['units']
            #     self.rnn_layers = params['rnn']['layers']
            #     self.rnn_name = params['rnn']['name']
            #     self.rnn_ln = params['rnn'].get('layer_norm', False)
            #     self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
            #     self.rnn_concat_input = params['rnn'].get(
            #         'concat_input', False)

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

            return

    def build(self, name, **kwargs):
        net = customNetBuilder.Network(self.params, **kwargs)
        return net
