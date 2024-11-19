"""pref-acc training types."""

from typing import NamedTuple

from brax.training.acme.types import NestedArray


class Transition(NamedTuple):
  """Container for a transition."""
  observation: NestedArray
  action: NestedArray
  reward: NestedArray
  true_reward: NestedArray
  discount: NestedArray
  next_observation: NestedArray
  extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray


class PreferencePair(NamedTuple):
  """Container for a preference pair."""
  segment1: Transition
  segment2: Transition
  preference: NestedArray