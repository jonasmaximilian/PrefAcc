"""Reward model tests."""

from absl.testing import absltest
import jax
from prefacc.training.reward_model import reward_model as rm


class RewardModelTest(absltest.TestCase):
  """Tests for reward model module."""

  def testRewardModelCreation(self):
    """Test reward model creation."""
    obs_size = 3
    action_size = 2
    model = rm.make_reward_model_network(obs_size, action_size)
    self.assertIsNotNone(model)

  def testRewardModelInference(self):
    """Test reward model inference."""
    obs_size = 3
    action_size = 2
    model = rm.make_reward_model_network(obs_size, action_size)
    params = model.init(jax.random.PRNGKey(0))
    obs = jax.random.uniform(jax.random.PRNGKey(0), (1, obs_size))
    action = jax.random.uniform(jax.random.PRNGKey(0), (1, action_size))
    reward = model.apply(params, obs, action)
    self.assertEqual(reward.shape, (1,))


if __name__ == '__main__':
  absltest.main()