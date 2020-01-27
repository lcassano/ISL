# ISL implementation
# Author: Lucas Cassano
# ======================================================================================================================
"""A simple implementation of ISL.
References:
1. "ISL: Optimal Policy Learning With Optimal Exploration-Exploitation Trade-Off" (Cassano et al., 2019)

"""

# Import all packages
from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env

import numpy as np
import sonnet as snt
import tensorflow as tf
from trfl.indexing_ops import batched_index
from trfl.target_update_ops import update_target_variables
from typing import Sequence


class ISL(base.Agent):
  """Information seeking learner."""

  def __init__(
      self,
      obs_spec: dm_env.specs.Array,
      action_spec: dm_env.specs.BoundedArray,
      q_network: snt.AbstractModule,
      target_q_network: snt.AbstractModule,
      rho_network: snt.AbstractModule,
      l_network: Sequence[snt.AbstractModule],
      target_l_network: Sequence[snt.AbstractModule],
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer_primal: tf.train.Optimizer,
      optimizer_dual: tf.train.Optimizer,
      optimizer_l: tf.train.Optimizer,
      learn_iters: int,
      l_approximators: int,
      min_l: float,
      kappa: float,
      eta1: float,
      eta2: float,
      seed: int = None,
  ):
    """Information seeking learner."""
    # ISL configurations.
    self.q_network = q_network
    self._target_q_network = target_q_network
    self.rho_network = rho_network
    self.l_network = l_network
    self._target_l_network = target_l_network
    self._num_actions = action_spec.maximum - action_spec.minimum + 1
    self._obs_shape = obs_spec.shape
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._optimizer_primal = optimizer_primal
    self._optimizer_dual = optimizer_dual
    self._optimizer_l = optimizer_l
    self._min_replay_size = min_replay_size
    self._replay = replay.Replay(capacity=replay_capacity) #ISLReplay(capacity=replay_capacity, average_l=0, mu=0)  #
    self._rng = np.random.RandomState(seed)
    tf.set_random_seed(seed)
    self._kappa = kappa
    self._min_l = min_l
    self._eta1 = eta1
    self._eta2 = eta2
    self._learn_iters = learn_iters
    self._l_approximators = l_approximators
    self._total_steps = 0
    self._total_episodes = 0
    self._learn_iter_counter = 0

    # Making the tensorflow graph
    o = tf.placeholder(shape=obs_spec.shape, dtype=obs_spec.dtype)
    q = q_network(tf.expand_dims(o, 0))
    rho = rho_network(tf.expand_dims(o, 0))
    l = []
    for k in range(self._l_approximators):
      l.append(tf.concat([l_network[k][a](tf.expand_dims(o, 0)) for a in range(self._num_actions)], axis=1))

    # Placeholders = (obs, action, reward, discount, next_obs)
    o_tm1 = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    a_tm1 = tf.placeholder(shape=(None,), dtype=action_spec.dtype)
    r_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    d_t = tf.placeholder(shape=(None,), dtype=tf.float32)
    o_t = tf.placeholder(shape=(None,) + obs_spec.shape, dtype=obs_spec.dtype)
    chosen_l = tf.placeholder(shape=1, dtype=tf.int32, name='chosen_l_tensor')

    q_tm1 = q_network(o_tm1)
    rho_tm1 = rho_network(o_tm1)
    train_q_value = batched_index(q_tm1, a_tm1)
    train_rho_value = batched_index(rho_tm1, a_tm1)
    train_rho_value_no_grad = tf.stop_gradient(train_rho_value)
    if self._target_update_period > 1:
      q_t = target_q_network(o_t)
    else:
      q_t = q_network(o_t)

    l_tm1_all = tf.stack([tf.concat([self.l_network[k][a](o_tm1) for a in range(self._num_actions)], axis=1) for k in range(self._l_approximators)], axis=-1)
    l_tm1 = tf.squeeze(tf.gather(l_tm1_all, chosen_l, axis=-1), axis=-1)
    train_l_value = batched_index(l_tm1, a_tm1)

    if self._target_update_period > 1:
      l_online_t_all = tf.stack([tf.concat([self.l_network[k][a](o_t) for a in range(self._num_actions)], axis=1) for k in range(self._l_approximators)], axis=-1)
      l_online_t = tf.squeeze(tf.gather(l_online_t_all, chosen_l, axis=-1), axis=-1)
      l_t_all = tf.stack([tf.concat([self._target_l_network[k][a](o_t) for a in range(self._num_actions)], axis=1) for k in range(self._l_approximators)], axis=-1)
      l_t = tf.squeeze(tf.gather(l_t_all, chosen_l, axis=-1), axis=-1)
      max_ind = tf.math.argmax(l_online_t, axis=1)
    else:
      l_t_all = tf.stack([tf.concat([self.l_network[k][a](o_t) for a in range(self._num_actions)], axis=1) for k in range(self._l_approximators)], axis=-1)
      l_t = tf.squeeze(tf.gather(l_t_all, chosen_l, axis=-1), axis=-1)
      max_ind = tf.math.argmax(l_t, axis=1)

    soft_max_value = tf.stop_gradient(tf.py_function(func=self.soft_max, inp=[q_t, l_t], Tout=tf.float32))
    q_target_value = r_t + discount*d_t*soft_max_value
    delta_primal = train_q_value - q_target_value
    loss_primal = tf.add(eta2*train_rho_value_no_grad*delta_primal, (1-eta2)*0.5*tf.square(delta_primal), name='loss_q')

    delta_dual = tf.stop_gradient(delta_primal)
    loss_dual = tf.square(delta_dual - train_rho_value, name='loss_rho')

    l_greedy_estimate = tf.add((1 - eta1) * tf.math.abs(delta_primal), eta1 * tf.math.abs(train_rho_value_no_grad),
                                 name='l_greedy_estimate')
    l_target_value = tf.stop_gradient(l_greedy_estimate + discount * d_t * batched_index(l_t, max_ind), name='l_target')
    loss_l = 0.5 * tf.square(train_l_value - l_target_value)

    train_op_primal = self._optimizer_primal.minimize(loss_primal)
    train_op_dual = self._optimizer_dual.minimize(loss_dual)
    train_op_l = self._optimizer_l.minimize(loss_l)

    # create target update operations
    if self._target_update_period > 1:
      target_updates = []
      target_update = update_target_variables(
        target_variables=self._target_q_network.get_all_variables(),
        source_variables=self.q_network.get_all_variables(),
      )
      target_updates.append(target_update)
      for k in range(self._l_approximators):
        for a in range(self._num_actions):
          model = self.l_network[k][a]
          target_model = self._target_l_network[k][a]
          target_update = update_target_variables(
            target_variables=target_model.get_all_variables(),
            source_variables=model.get_all_variables(),
          )
          target_updates.append(target_update)

    # Make session and callables.
    session = tf.Session()
    self._sgd = session.make_callable([train_op_l, train_op_primal, train_op_dual], [o_tm1, a_tm1, r_t, d_t, o_t, chosen_l])
    self._q_fn = session.make_callable(q, [o])
    self._rho_fn = session.make_callable(rho, [o])
    self._l_fn = []
    for k in range(self._l_approximators):
      self._l_fn.append(session.make_callable(l[k], [o]))
    if self._target_update_period > 1:
      self._update_target_nets = session.make_callable(target_updates)
    session.run(tf.global_variables_initializer())

  def policy(self, timestep: dm_env.TimeStep) -> base.Action:
    """Select actions according to optimal exploration-exploitation trade-off."""

    chosen_l = np.random.randint(self._l_approximators)
    q_values = self._q_fn(timestep.observation)
    l_values = self._l_fn[chosen_l](timestep.observation)

    if self._total_steps < self._min_replay_size:
      action = self._rng.randint(self._num_actions)
      return action

    ordered_a, probs = self.unnormalized_probabilities(q_values[0], l_values[0])
    probs = np.divide(probs, np.sum(probs))  # Normalize
    if ordered_a.size > 1:
      rand_number = self._rng.rand()
      action_index = next(x for x, val in enumerate(np.cumsum(probs)) if val > rand_number)  # Sample.
    else:
      action_index = 0

    action = int(ordered_a[action_index])
    distribution = np.zeros(self._num_actions)
    distribution[ordered_a] = probs

    return action

  def soft_max(self, q_values, l_values):
    """Calculate the soft-max."""

    np_q_values = q_values.numpy()
    np_l_values = l_values.numpy()

    soft_maxes = np.empty(q_values.shape[0])
    for k, (q_vec, l_vec) in enumerate(zip(np_q_values, np_l_values)):
      _, probs = self.unnormalized_probabilities(q_vec, l_vec)
      soft_maxes[k] = np.sum(probs)/np.max(l_vec)

    soft_max_val = self._kappa*np.log(soft_maxes), np.amax(np_q_values, axis=1)

    if np.isinf(soft_max_val).any():
      return np.amax(np_q_values, axis=1)  # Save against numerical issues
    elif np.isnan(soft_max_val).any():
      return np.amax(np_q_values, axis=1)  # Save against numerical issues
    else:
      return soft_max_val

  def unnormalized_probabilities(self, q_values, l_values):
    """Calculate unnormalized probabilities according to optimal exploration-exploitation strategy."""

    if q_values.size == 1:  # Trivial case
        return 0, q_values

    ordered_a = self.pareto_actions(q_values, l_values)  # Obtain set of ordered pareto optimal actions,
    ordered_q = q_values[ordered_a]                      # ordered_smaller to bigger l.
    ordered_l = l_values[ordered_a]

    ql_prods = ordered_q * ordered_l
    delta_ql = np.divide(ql_prods - np.append(0, ql_prods[0:-1]), ordered_l - np.append(0, ordered_l[0:-1])+1e-8)
    exp_delta = np.exp(delta_ql/self._kappa)
    diff_exp = exp_delta - np.append(exp_delta[1:], 0)
    unnorm_probs = diff_exp * ordered_l
    return ordered_a, unnorm_probs

  @staticmethod
  def pareto_actions(q: np.array, l: np.array) -> np.array:
    """Finds the Pareto optimal actions"""
    ordered_l_indices = np.argsort(l)[::-1]
    ordered_q_indices = np.argsort(q)

    ordered_q_indices_opt = ordered_q_indices
    opt_index = 0
    read_index = 0

    # Eliminate Pareto dominated actions
    for n, l_val_ind in enumerate(ordered_l_indices):
      for k, q_val_ind in enumerate(ordered_q_indices[read_index:]):
        if l_val_ind == q_val_ind:
          if (n < ordered_l_indices.size-1) and (k < ordered_q_indices[read_index:].size-1) and (l[l_val_ind] == l[ordered_q_indices[read_index+k+1]]):
            read_index += k + 1
            break  # There's two l's which are equal, therefore skip the one with the lowest q value.
          ordered_q_indices_opt[opt_index] = q_val_ind
          opt_index += 1
          read_index += k + 1
          break

    ordered_q_indices = ordered_q_indices_opt[:opt_index]
    ordered_l_indices = ordered_q_indices[::-1]

    # Eliminate mixed pareto dominated actions
    if ordered_q_indices.size > 2:
      q = q[ordered_l_indices]
      l = l[ordered_l_indices]
      lq = q*l

      r_index_l_l = l.size - 1  # large l
      r_index_m_l = l.size - 2  # medium l
      r_index_s_l = l.size - 3  # small l
      old_r_index_l_l = l.size - 1  # Initialized as r_index_l_l
      w_index = 1
      while r_index_m_l > 0:
        if (lq[r_index_m_l]-lq[r_index_s_l])*(l[r_index_l_l]-l[r_index_m_l]) > (lq[r_index_l_l]-lq[r_index_m_l])*(l[r_index_m_l]-l[r_index_s_l]):
          ordered_q_indices[w_index] = ordered_l_indices[r_index_m_l]
          w_index += 1
          r_index_l_l = r_index_m_l
          r_index_m_l = r_index_s_l
          r_index_s_l -= 1
        else:  # Mixed Pareto dominated
          if w_index > 1:
            r_index_m_l = r_index_l_l
            r_index_l_l = old_r_index_l_l
            w_index -= 1
          else:
            r_index_m_l = r_index_s_l
            r_index_s_l -= 1
      ordered_q_indices[w_index] = ordered_l_indices[r_index_m_l]
      w_index += 1
      return ordered_q_indices[np.arange(w_index-1, -1, -1)]
    else:
      return ordered_l_indices

  def update(self, old_step: dm_env.TimeStep, action: base.Action,
             new_step: dm_env.TimeStep):
    """Takes in a transition from the environment."""

    # Add this transition to replay.
    sample = [
        old_step.observation,
        action,
        new_step.reward,
        new_step.discount,
        new_step.observation
    ]
    self._replay.add(sample)

    self._total_steps += 1

    if self._total_steps % self._sgd_period != 0:
      return

    if self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD and SGA on primal and dual variables, respectively.
    for _ in range(self._learn_iters):
      minibatch = self._replay.sample(self._batch_size)
      self._sgd(*minibatch, [np.random.randint(self._l_approximators)])
      if (self._target_update_period > 1) and (self._learn_iter_counter % self._target_update_period == 0):
        self._update_target_nets()
