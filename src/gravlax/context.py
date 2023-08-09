from __future__ import annotations

import timeit
import typing as tp
from types import TracebackType

import jax.numpy as jnp

from .tree_utils import reduce_pytree, tree_stack
from .types import PyTree


def prepend_to_key(d, prefix):
    return {f'{prefix}{k}': v for k, v in d.items()}


class BatchManager:

    def __init__(self, name: str, n_batches=10):

        self.name = name
        self.n_batches = n_batches

        self.start_time: float
        self._loss_list = []

    @property
    def time_elapsed(self):
        return timeit.default_timer() - self.start_time

    def register_loss(self, loss_dict: dict[str, PyTree]):
        self._loss_list.append(loss_dict)

    def reduce(self, op: tp.Callable[..., PyTree] = jnp.mean, **op_kwargs: dict[str, object]):
        return prepend_to_key(reduce_pytree(tree_stack(self._loss_list), op, **op_kwargs), prefix=f'{self.name}_')

    def __enter__(self) -> BatchManager:
        self.start_time = timeit.default_timer()
        return self

    @tp.overload
    def __exit__(self, exc_type: type[BaseException], exc_val: BaseException, exc_tb: TracebackType) -> None:
        ...

    @tp.overload
    def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
        ...

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:

        self.time = self.time_elapsed

        if not (n := len(self._loss_list)) == self.n_batches:
            raise ValueError(f'Only recorded {n}/{self.n_batches} batches.')
