from typing import Sequence, Tuple

from brax.training import networks
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
from prefacc.training.reward_model import reward_model

@flax.struct.dataclass
class RewardModelNetworks:
    reward_model_network: networks.FeedForwardNetwork


def make_inference_fn(reward_model_networks: RewardModelNetworks):
    """ Creates params and inference function for the reward model """

    def make_reward_model(params):

        reward_model_network = reward_model_networks.reward_model_network

        def reward_model(observations: types.Observation, actions: types.Action) -> float:
            logits = reward_model_network.apply(params, observations, actions)
            return logits

        return reward_model

    return make_reward_model


def make_reward_model_networks(
    observation_size: int,
    action_size: int,
    hidden_layer_sizes: Sequence[int] = (128, 128),
) -> RewardModelNetworks:
    reward_model_network = reward_model.make_reward_model_network(
        obs_size=observation_size,
        action_size=action_size,
        hidden_layer_sizes=hidden_layer_sizes
    )
    return RewardModelNetworks(reward_model_network=reward_model_network)
    