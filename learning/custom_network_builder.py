from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np


class customNetBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        print(network_builder)
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            print('INPUT SHAPE')
            print(input_shape)
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_fusion_mlp = nn.Sequential()
            self.critic_fusion_mlp = nn.Sequential()

            # DEFINE CNN
            self.grid_size = self.cnn['grid_size']
            cnn_input_shape = (1, self.grid_size, self.grid_size)
            cnn_args = {
                'ctype': self.cnn['type'],
                'input_shape': cnn_input_shape,
                'convs': self.cnn['convs'],
                'activation': self.cnn['activation'],
                'norm_func_name': self.normalization,
            }
            self.actor_cnn = self._build_conv(**cnn_args)
            if self.separate:
                self.critic_cnn = self._build_mlp(**mlp_args)

            # DEFINE MLP
            mlp_input_shape = (input_shape[0]-self.grid_size**2)
            mlp_out_size = self.mlp_units[-1]
            mlp_args = {
                'input_size': mlp_input_shape,
                'units': self.mlp_units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            # DEFINE FUSION MLP
            fusion_mlp_input_shape = self._calc_input_size(
                cnn_input_shape, self.actor_cnn)
            fusion_mlp_input_shape += mlp_out_size
            print('fusion input shape:')
            print(fusion_mlp_input_shape)
            out_size = self.fusion_mlp_units[-1]
            fusion_mlp_args = {
                'input_size': fusion_mlp_input_shape,
                'units': self.fusion_mlp_units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            self.actor_fusion_mlp = self._build_mlp(**fusion_mlp_args)
            if self.separate:
                self.critic_fusion_mlp = self._build_mlp(**fusion_mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(
                self.value_activation)
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            '''
                for multidiscrete actions num is a tuple
            '''
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList(
                    [torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(
                    self.space_config['mu_activation'])
                mu_init = self.init_factory.create(
                    **self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(
                    self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(
                    **self.space_config['sigma_init'])

                if self.space_config['fixed_sigma']:
                    self.sigma = nn.Parameter(torch.zeros(
                        actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            cnn_init = self.init_factory.create(**self.cnn['initializer'])

            print('MODULES:')
            for m in self.modules():
                print(m)
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.space_config['fixed_sigma']:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            seq_length = obs_dict.get('seq_length', 1)
            states = obs_dict.get('rnn_states', None)
            cnn_input = obs[:, :self.grid_size **
                            2].view(-1, 1, self.grid_size, self.grid_size)
            mlp_input = obs[:, self.grid_size**2:]
            if self.separate:
                print('not supported!')
                pass
            else:
                out_mlp = mlp_input
                out_mlp = self.actor_mlp(out_mlp)

                out_cnn = cnn_input
                out_cnn = self.actor_cnn(out_cnn)
                out_cnn = out_cnn.flatten(1)

                out = torch.cat((out_mlp, out_cnn), dim=-1)
                out = self.actor_fusion_mlp(out)

                value = self.value_act(self.value(out))

                if self.central_value:
                    return value, states
                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.space_config['fixed_sigma']:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu*0 + sigma, value, states

        def is_separate_critic(self):
            return self.separate

        def is_rnn(self):
            return self.has_rnn

        def load(self, params):
            self.separate = params.get('separate', False)
            self.mlp_units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get(
                'norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_rnn = 'rnn' in params
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get(
                'joint_obs_actions', None)

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

            if 'cnn' in params:
                self.has_cnn = True
                self.cnn = params['cnn']
            else:
                self.has_cnn = False

            if 'fusion_mlp' in params:
                self.has_fusion_mlp = True
                self.fusion_mlp_units = params['fusion_mlp']['units']
            else:
                self.has_fusion_mlp = False

            return

    def build(self, name, **kwargs):
        net = customNetBuilder.Network(self.params, **kwargs)
        return net
