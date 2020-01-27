# ISL implementation
# Author: Lucas Cassano
# Paper: "ISL: Optimal Policy Learning With Optimal Exploration-Exploitation Trade-Off" (Cassano et al., 2019)
# ======================================================================================================================
"""Run agent on a bsuite experiment."""

# Import all packages

import os
from absl import app
from absl import flags

from bsuite import bsuite
from bsuite import sweep

from bsuite.baselines import experiment
from bsuite.baselines.isl import isl
from bsuite.baselines.utils import pool

import sonnet as snt
import tensorflow as tf
from typing import Text


# bsuite logging
flags.DEFINE_string('bsuite_id', 'deep_sea_stochastic/0',
                    'specify either a single bsuite_id (e.g. catch/0)\n'
                    'or a global variable from bsuite.sweep (e.g. SWEEP for '
                    'all of bsuite, or DEEP_SEA for just deep_sea experiment).')
flags.DEFINE_string('save_path', '/Users/lucas/Documents/bsuite/bsuite/baselines/isl/results', 'where to save bsuite results')
flags.DEFINE_enum('logging_mode', 'terminal', ['csv', 'sqlite', 'terminal'],
                  'which form of logging to use for bsuite results')
flags.DEFINE_boolean('overwrite', True, 'overwrite csv logging if found')
flags.DEFINE_integer('num_episodes', None, 'Overrides number of training eps.')

# Network options
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_integer('l_approximators', 2, 'number of NN approximators for l values')

# Core ISL options
flags.DEFINE_integer('batch_size', 256, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', 0.99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 512, 'min transitions for sampling')
flags.DEFINE_integer('sgd_period', 10, 'environment steps between net updates')
flags.DEFINE_integer('target_update_period', 2, 'steps between target net updates')
flags.DEFINE_float('q_learning_rate', 1e-4, 'learning rate for q network')
flags.DEFINE_float('rho_learning_rate', 1e-4, 'learning rate for rho network')
flags.DEFINE_float('l_learning_rate', 1e-4, 'learning rate for l network')
flags.DEFINE_float('min_l', 1e-12, 'minimum allowed uncertainty for l')
flags.DEFINE_float('max_l', 100, 'maximum allowed uncertainty for l')
flags.DEFINE_float('eta1', 1, 'Hyperparameter eta_1')
flags.DEFINE_float('eta2', 0.5, 'Hyperparameter eta_2')
flags.DEFINE_float('kappa', 1, 'KL divergence weighting parameter')
flags.DEFINE_integer('learn_iters', 1, 'gradient descent iterations for every sgd_period')
flags.DEFINE_integer('seed', 0, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')


FLAGS = flags.FLAGS


def run(bsuite_id: Text) -> Text:
  """Runs a ISL agent on a given bsuite environment."""

  env = bsuite.load_and_record(
    bsuite_id=bsuite_id,
    save_path=FLAGS.save_path,
    logging_mode=FLAGS.logging_mode,
    overwrite=FLAGS.overwrite,
  )

  # Making the NNs (q, rho and l).
  hidden_units = [FLAGS.num_units] * FLAGS.num_hidden_layers

  q_network = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP(hidden_units + [env.action_spec().num_values])])
  target_q_network = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP(hidden_units + [env.action_spec().num_values])])

  rho_network = snt.Sequential([snt.BatchFlatten(), snt.nets.MLP(hidden_units + [env.action_spec().num_values])])

  l_network = [[None for _ in range(env.action_spec().num_values)] for _ in range(FLAGS.l_approximators)]
  target_l_network = [[None for _ in range(env.action_spec().num_values)] for _ in range(FLAGS.l_approximators)]
  for k in range(FLAGS.l_approximators):
    for a in range(env.action_spec().num_values):
      l_network[k][a] = snt.Sequential([
        snt.BatchFlatten(),
        snt.nets.MLP(hidden_units, activate_final=True, initializers={'b': tf.constant_initializer(0)}),
        snt.Linear(1, initializers={'b': tf.constant_initializer(0)}),
        lambda x: (FLAGS.max_l-FLAGS.min_l)*tf.math.sigmoid(x) + FLAGS.min_l])

      target_l_network[k][a] = snt.Sequential([
        snt.BatchFlatten(),
        snt.nets.MLP(hidden_units, activate_final=True, initializers={'b': tf.constant_initializer(0)}),
        snt.Linear(1, initializers={'b': tf.constant_initializer(0)}),
        lambda x: (FLAGS.max_l-FLAGS.min_l)*tf.math.sigmoid(x) + FLAGS.min_l])

  agent = isl.ISL(
    obs_spec=env.observation_spec(),
    action_spec=env.action_spec(),
    q_network=q_network,
    target_q_network=target_q_network,
    rho_network=rho_network,
    l_network=l_network,
    target_l_network=target_l_network,
    batch_size=FLAGS.batch_size,
    discount=FLAGS.agent_discount,
    replay_capacity=FLAGS.replay_capacity,
    min_replay_size=FLAGS.min_replay_size,
    sgd_period=FLAGS.sgd_period,
    target_update_period=FLAGS.target_update_period,
    optimizer_primal=tf.train.AdamOptimizer(learning_rate=FLAGS.q_learning_rate),
    optimizer_dual=tf.train.AdamOptimizer(learning_rate=FLAGS.rho_learning_rate),
    optimizer_l=tf.train.AdamOptimizer(learning_rate=FLAGS.l_learning_rate),
    learn_iters=FLAGS.learn_iters,
    l_approximators=FLAGS.l_approximators,
    min_l=FLAGS.min_l,
    kappa=FLAGS.kappa,
    eta1=FLAGS.eta1,
    eta2=FLAGS.eta2,
    seed=FLAGS.seed
  )

  experiment.run(
    agent=agent,
    environment=env,
    num_episodes=FLAGS.num_episodes or env.bsuite_num_episodes,
    verbose=FLAGS.verbose)

  return bsuite_id


def main(argv):
  """Parses whether to run a single bsuite_id, or multiprocess sweep."""
  del argv
  bsuite_id = FLAGS.bsuite_id

  if bsuite_id in sweep.SWEEP:
    print('Running a single bsuite_id={}'.format(bsuite_id))
    run(bsuite_id)

  elif hasattr(sweep, bsuite_id):
    bsuite_sweep = getattr(sweep, bsuite_id)
    print('Running a sweep over bsuite_id in sweep.{}'.format(bsuite_sweep))
    FLAGS.verbose = False
    pool.map_mpi(run, bsuite_sweep)

  else:
    raise ValueError('Invalid flag bsuite_id={}'.format(bsuite_id))


if __name__ == '__main__':
  app.run(main)
