import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import torch.optim as optim  # type: ignore
from torch.autograd import Variable  # type: ignore

import pyro
import pyro.distributions as dist
from pyro.infer import SVI
from pyro.optim import ClippedAdam
# from pyro.distributions.transformed_distribution import TransformedDistribution

import os
import sys
import time
import argparse
import itertools
from os.path import join, exists
import six.moves.cPickle as pickle

class BernoulliEmitter(nn.Module):
    def __init__(self, obs_dim, num_classes, latent_dim, emission_dim):
        super(CategoricalEmitter, self).__init__()
        self.lin_gate_latent_to_hidden = nn.Linear(latent_dim, emission_dim)
        self.lin_gate_hidden_to_observed = nn.Linear(emission_dim, obs_dim)

        self.lin_prop_latent_to_hidden = nn.Linear(
                                            latent_dim,
                                            emission_dim
                                            )

        self.lin_prop_hidden_to_hidden = nn.Linear(
                                            emission_dim,
                                            emission_dim)
        self.lin_prop_hidden_to_props = nn.Linear(
                                            emission_dim,
                                            obs_dim)
        self.lin_props = nn.Linear(
                            num_classes * obs_dim,
                            num_classes * obs_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.num_classes = num_classes
        self.obs_dim = obs_dim

    def forward(self, z_t):
        h1 = self.lin_prop_latent_to_hidden(z_t)
        h2 = self.lin_prop_hidden_to_hidden(h1)

        props = self.lin_props(self.relu(self.lin_prop_hidden_to_props(h2)))
        props = self.softplus(props)

        return props

class CategoricalEmitter(nn.Module):
    def __init__(self, obs_dim, num_classes, latent_dim, emission_dim):
        super(CategoricalEmitter, self).__init__()
        self.lin_gate_latent_to_hidden = nn.Linear(latent_dim, emission_dim)
        self.lin_gate_hidden_to_observed = nn.Linear(emission_dim, obs_dim)

        self.lin_prop_latent_to_hidden = nn.Linear(
                                            latent_dim,
                                            num_classes * emission_dim
                                            )

        self.lin_prop_hidden_to_hidden = nn.Linear(
                                            num_classes * emission_dim,
                                            num_classes * emission_dim)
        self.lin_prop_hidden_to_props = nn.Linear(
                                            num_classes * emission_dim,
                                            num_classes * obs_dim)
        self.lin_props = nn.Linear(
                            num_classes * obs_dim,
                            num_classes * obs_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        self.num_classes = num_classes
        self.obs_dim = obs_dim

    def forward(self, z_t):
        h1 = self.lin_prop_latent_to_hidden(z_t)
        h2 = self.lin_prop_hidden_to_hidden(h1)

        props = self.lin_props(self.relu(self.lin_prop_hidden_to_props(h2)))
        props = self.softmax(props.view(-1, self.obs_dim, self.num_classes))

        return props

class GaussianEmitter(nn.Module):
    def __init__(self, obs_dim, latent_dim, emission_dim):
        super(GaussianEmitter, self).__init__()
        self.lin_gate_latent_to_hidden = nn.Linear(latent_dim, emission_dim)
        self.lin_gate_hidden_to_observed = nn.Linear(emission_dim, obs_dim)

        self.lin_prop_latent_to_hidden = nn.Linear(latent_dim, emission_dim)

        self.lin_prop_hidden_to_hidden = nn.Linear(emission_dim, emission_dim)
        self.lin_prop_hidden_to_mu = nn.Linear(emission_dim, obs_dim)

        self.lin_latent_to_mu = nn.Linear(latent_dim, obs_dim)
        self.lin_sig = nn.Linear(obs_dim, obs_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_t):
        h1 = self.lin_prop_latent_to_hidden(z_t)
        h2 = self.lin_prop_hidden_to_hidden(h1)

        mu = self.lin_prop_hidden_to_mu(h2)

        sigma = self.softplus(self.lin_sig(self.relu(mu)))

        return mu, sigma


class Transferer(nn.Module):
    def __init__(self, latent_dim, transfer_dim):
        super(Transferer, self).__init__()
        self.lin_gate_z_to_hidden = nn.Linear(latent_dim, transfer_dim)
        self.lin_gate_hidden_to_z = nn.Linear(transfer_dim, latent_dim)

        self.lin_proposed_mean_z_to_hidden = nn.Linear(latent_dim, transfer_dim)
        self.lin_proposed_mean_hidden_to_z = nn.Linear(transfer_dim, latent_dim)
        self.lin_sig = nn.Linear(latent_dim, latent_dim)
        self.lin_z_to_mu = nn.Linear(latent_dim, latent_dim)

        self.lin_z_to_mu.weight.data = torch.eye(latent_dim)
        self.lin_z_to_mu.bias.data = torch.zeros(latent_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1):
        gate_intermediate = self.relu(self.lin_gate_z_to_hidden(z_t_1))
        gate = self.sigmoid(self.lin_gate_hidden_to_z(gate_intermediate))

        proposed_mean_intermediate = self.relu(self.lin_proposed_mean_z_to_hidden(z_t_1))
        proposed_mean = self.lin_proposed_mean_hidden_to_z(proposed_mean_intermediate)

        mu = (1 - gate) * self.lin_z_to_mu(z_t_1) + gate * proposed_mean
        sigma = self.softplus(self.lin_sig(self.relu(proposed_mean)))
        return mu, sigma


class Combiner(nn.Module):
    def __init__(self, latent_dim, combiner_dim):
        super(Combiner, self).__init__()
        self.lin_z_to_hidden = nn.Linear(latent_dim, combiner_dim)
        self.lin_hidden_to_mu = nn.Linear(combiner_dim, latent_dim)
        self.lin_hidden_to_sigma = nn.Linear(combiner_dim, latent_dim)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, z_t_1, h_l_rnn, h_r_rnn):
        h_combined = float(1./3.) * (self.tanh(self.lin_z_to_hidden(z_t_1)) + h_l_rnn + h_r_rnn)
        mu = self.lin_hidden_to_mu(h_combined)
        sigma = self.softplus(self.lin_hidden_to_sigma(h_combined))
        return mu, sigma


class DMM(nn.Module):
    def __init__(self, obs_dim, emit_dist, latent_dim=50, emission_dim=50,
                 transfer_dim=30, combiner_dim=50, rnn_dim=200, rnn_dropout_rate=0.0, num_classes=0,  use_cuda=False):
        super(DMM, self).__init__()
        self.emit_dist = emit_dist

        if self.emit_dist == 'categorical':
            self.num_classes = num_classes
            self.emitter = CategoricalEmitter(obs_dim, num_classes, latent_dim, emission_dim)

        elif self.emit_dist == 'gaussian':
            self.emitter = GaussianEmitter(obs_dim, latent_dim, emission_dim)

        elif self.emit_dist == 'bernoulli':
            self.emitter = BernoulliEmitter(obs_dim, num_classes, latent_dim, emission_dim)


        self.trans = Transferer(latent_dim, transfer_dim)
        self.combiner = Combiner(latent_dim, rnn_dim)
        rnn_input_dim = obs_dim if self.emit_dist != 'categorical' else obs_dim * num_classes
        self.rnn = nn.RNN(input_size=rnn_input_dim, hidden_size=rnn_dim, nonlinearity='relu',
                          batch_first=True, bidirectional=False, num_layers=1,
                          dropout=rnn_dropout_rate)

        self.z_0 = nn.Parameter(torch.zeros(latent_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(latent_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, rnn_dim))

        self.prev_timing_distributions = None

        self.use_cuda = use_cuda
        if use_cuda:
            self.cuda()

    def model(self, mini_batch, mini_batch_reversed):

        T_max = mini_batch.size(1)
        pyro.module("dmm", self)

        z_prev = self.z_0.expand(mini_batch.size(0), self.z_0.size(0))

        for t in range(1, T_max + 1):
            z_mu, z_sigma = self.trans(z_prev)
            z_t = pyro.sample("z_%d" % t,
                              dist.normal,
                              z_mu,
                              z_sigma)

            if self.emit_dist == 'categorical':
                emission_probs_t = self.emitter(z_t)
                timings = pyro.sample("obs_x_%d" % t,
                            dist.categorical,
                            emission_probs_t,
                            one_hot=False,
                            obs=mini_batch[:, t - 1, :])

            elif self.emit_dist == 'bernoulli':
                emission_mus_t, emission_sigmas_t = self.emitter(z_t)
            elif self.emit_dist == 'gaussian':
                emission_mus_t, emission_sigmas_t = self.emitter(z_t)

                timings = pyro.sample("obs_x_%d" % t,
                            dist.normal,
                            emission_mus_t,
                            emission_sigmas_t,
                            obs=mini_batch[:, t - 1, :])
            z_prev = z_t
        return z_prev


    def guide(self, mini_batch, mini_batch_reversed):
        T_max = mini_batch.size(1)
        pyro.module("dmm", self)

        h_0_contig = self.h_0.expand(1, mini_batch.size(0), self.rnn.hidden_size).contiguous()
        if self.emit_dist == 'categorical':
            mini_batch = mini_batch.view(mini_batch.shape[0], mini_batch.shape[1], -1)

        print(mini_batch.shape)
        import pdb; pdb.set_trace()
        left, h_0 = self.rnn(mini_batch, h_0_contig)
        right, h_0 = self.rnn(mini_batch_reversed, h_0_contig)

        z_prev = self.z_q_0

        for t in range(1, T_max + 1):
            z_mu, z_sigma = self.combiner(z_prev, left[:, t - 1, :], right[:, T_max - 1, :])
            z_dist = dist.normal

            z_t = pyro.sample("z_%d" % t,
                              z_dist,
                              z_mu,
                              z_sigma)
            z_prev = z_t
        return z_prev


    def predict(self, test_sequence, test_sequence_reversed):
        num_samples = 1
        # run the guide to get the posterior p(
        # this thing actually samples from te guide, but gives the log weight
        # based on the pointwise calculation of the original model
        posterior = pyro.infer.Importance(self.model, self.guide, num_samples=num_samples)
        #  return posterior

        a = posterior._traces(test_sequence, test_sequence_reversed)
        (model_trace, weight) = next(a)
        z_t = model_trace.nodes["_RETURN"]["value"]

        z_mu, z_sigma = self.trans(z_t)
        z_tp1 = pyro.sample("z_t_plus_1",
                          dist.normal,
                          z_mu,
                          z_sigma)

        emission_mus_t, emission_sigmas_t = self.emitter(z_tp1)
        zeros = Variable(torch.zeros(emission_mus_t.size()))
        timings = pyro.sample("obs_t_plus_1",
                    dist.normal,
                    emission_mus_t,
                    emission_sigmas_t
                    )

        self.prev_timing_distributions = (emission_mus_t, emission_sigmas_t)

        return timings
