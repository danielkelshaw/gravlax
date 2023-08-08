from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from .types import PyTree


def tree_stack(trees: Sequence[PyTree]) -> PyTree:
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


def tree_unstack(tree: PyTree) -> list[PyTree]:
    leaves, treedef = jtu.tree_flatten(tree)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]


def reduce_pytree(pytree: PyTree, op: Callable[..., jax.Array] = jnp.mean, op_kwargs: dict[str, Any] | None = None) -> PyTree:

    if not op_kwargs:
        op_kwargs = {}

    return jtu.tree_map(lambda x: op(x, **op_kwargs), pytree)
