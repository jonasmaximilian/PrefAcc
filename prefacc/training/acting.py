"""pref-acc training acting functions.

This file is modified from the original file in Google Brax
(https://github.com/google/brax/blob/main/brax/training/acting.py).
Original work Copyright 2024 The Brax Authors.
Licensed under the Apache License, Version 2.0.

Modifications Copyright 2024 Senft-Raiß
Licensed under the MIT License.
"""

from typing import Sequence, Tuple, Union

from brax import envs
from brax.training.types import Policy
from brax.training.types import PRNGKey
from brax.v1 import envs as envs_v1
import jax
import numpy as np
from prefacc.training.types import Transition

State = Union[envs.State, envs_v1.State]
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    reward_model,
    key: PRNGKey,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect data."""
  actions, policy_extras = policy(env_state.obs, key)
  reward = reward_model(env_state.obs, actions)
  nstate = env.step(env_state, actions)
  state_extras = {x: nstate.info[x] for x in extra_fields}
  return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=env_state.obs,
      action=actions,
      reward=reward,
      true_reward=nstate.reward,
      discount=1 - nstate.done,
      next_observation=nstate.obs,
      extras={
          'policy_extras': policy_extras,
          'state_extras': state_extras
      })


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    reward_model,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = ()
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""

  @jax.jit
  def f(carry, unused_t):
    state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    nstate, transition = actor_step(
        env, state, policy, reward_model, current_key, extra_fields=extra_fields)
    return (nstate, next_key), transition

  (final_state, _), data = jax.lax.scan(
      f, (env_state, key), (), length=unroll_length)
  return final_state, data