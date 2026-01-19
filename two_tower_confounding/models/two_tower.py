from typing import Dict, Callable

import jax
import rax
from flax import nnx
from flax.struct import dataclass
from jax import Array
from rax._src.types import ReduceFn
import jax.numpy as jnp

from two_tower_confounding.utils import reduce_per_query


@dataclass
class TwoTowerOutput:
    click: Array
    examination: Array
    relevance: Array


class TwoTowerModel(nnx.Module):
    def __init__(
        self,
        relevance_tower: nnx.Module,
        bias_tower: nnx.Module,
        loss_fn: Callable = rax.pointwise_sigmoid_loss,
        reduce_fn: ReduceFn = reduce_per_query,
        use_propensity_weighting: bool = False,
        bias_norm_loss: float = 0.3,
    ):
        self.relevance_tower = relevance_tower
        self.bias_tower = bias_tower
        self.loss_fn = loss_fn
        self.reduce_fn = reduce_fn
        self.use_propensity_weighting = use_propensity_weighting
        self.bias_norm_loss = bias_norm_loss

    def __call__(self, batch: Dict) -> TwoTowerOutput:
        relevance = self.relevance_tower(batch)

        examination = self.bias_tower(batch)
        click = examination + relevance

        return TwoTowerOutput(
            relevance=relevance,
            examination=examination,
            click=click,
        )

    def compute_loss(self, output: TwoTowerOutput, batch: Dict) -> Array:
        weights = 1 / batch["propensities"] if self.use_propensity_weighting else None

        position_bias = self.bias_tower.get_position_bias()
        bias_mean_loss = jnp.mean(position_bias) ** 2

        return self.loss_fn(
                scores=output.click,
                labels=batch["clicks"],
                where=batch["mask"],
                weights=weights,
                reduce_fn=self.reduce_fn,
            )  + self.bias_norm_loss * bias_mean_loss

    def compute_test_loss(self, output: TwoTowerOutput, batch: Dict) -> Array:
        weights = 1 / batch["propensities"] if self.use_propensity_weighting else None

        return self.loss_fn(
                scores=output.click,
                labels=batch["clicks"],
                where=batch["mask"],
                weights=weights,
                reduce_fn=self.reduce_fn,
        )

    def predict_relevance(self, batch: Dict) -> Array:
        return self.relevance_tower(batch)

