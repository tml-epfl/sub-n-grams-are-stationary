import jax
from jax import numpy as jnp, vmap, jit, lax, random as jr, tree_util as jtu
from jax.numpy import linalg as jla
from jax.flatten_util import ravel_pytree
from jax.tree_util import register_pytree_node
from flax import linen as nn
import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from tqdm.auto import tqdm, trange
from collections import namedtuple
from functools import partial
from jaxopt.tree_util import *
from typing import (
    Any,
    Callable,
    Union,
    List,
)

from pathlib import Path

import optax
import tyro
from PIL import Image
import pickle
from simple_pytree import Pytree, static_field


def get_parameter_norms_dict(model):
    norms = {}
    param_info = _get_param_info(model)
    for name, norm in param_info:
        norms[name] = norm
    return norms


def _get_param_info(pytree, prefix=""):
    param_info = []
    for name, value in vars(pytree).items():
        if isinstance(value, jnp.ndarray):  # Base case: parameter array
            norm = jnp.linalg.norm(value)
            full_name = prefix + "." + name if prefix else name
            param_info.append((full_name, norm))
        elif isinstance(value, Pytree):  # Recursive step: Pytree attribute
            new_prefix = prefix + "." + name if prefix else name
            param_info.extend(_get_param_info(value, new_prefix))
        elif isinstance(
            value, list
        ):  # Handle lists (like layers in MLP, A, Q, Kt, V in transformers)
            for layer, item in enumerate(value):
                if isinstance(item, jnp.ndarray):  # Handle Pytrees or arrays in lists
                    for head in range(item.shape[0]):
                        param = item[head, :, :]
                        full_name = (
                            prefix
                            + "."
                            + name
                            + "_layer"
                            + f"{layer}_head"
                            + f"{head}"
                            + "_norm"
                            if prefix
                            else name + "_layer" + f"{layer}_head" + f"{head}" + "_norm"
                        )
                        norm = jnp.linalg.norm(param)
                        param_info.append((full_name, norm))

    return param_info


def generate_markov_dag(seq_len, lag, order):
    dag = [[] for _ in range(order + lag)]
    for i in range(order + lag, seq_len + 1):
        dag.append([i - j for j in range(1 + lag, order + lag + 1)])
    return dag


class RNG:
    def __init__(self, seed=None, key=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        elif key is not None:
            self.key = key
        else:
            raise Exception("RNG expects either a seed or random key.")

    def next(self, n_keys=1):
        if n_keys > 1:
            return jax.random.split(self.next(), n_keys)
        else:
            self.key, key = jax.random.split(self.key)
            return key

    def __getattr__(self, name):
        return partial(getattr(jax.random, name), self.next())


register_pytree_node(
    RNG,
    lambda rng: ((rng.key,), None),
    lambda _, c: RNG(key=c[0]),
)
