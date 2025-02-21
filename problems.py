from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence

import jax
from jax import jit, lax
from jax import numpy as jnp
from jax import random as jr
from jax import vmap
from jax.numpy import linalg as jla


def get_stationary(pi):
    mu = jla.svd(pi.T - jnp.eye(pi.shape[0]))[-1][-1]
    return mu / mu.sum()


class InContextTree:
    def __init__(self, vocab_size, dag, alpha):
        assert jnp.all(dag < jnp.arange(len(dag)))
        self.vocab_size = vocab_size
        self.dag = dag
        self.alpha = alpha

    def sample(self, key):
        pi_key, seq_key, test_key = jr.split(key, 3)
        prior = self.alpha * jnp.ones(self.vocab_size)
        pi = jr.dirichlet(pi_key, prior, [self.vocab_size])
        mu = get_stationary(pi)
        x = jnp.zeros((len(self.dag) + 1,), dtype=int)

        def step(i, carry):
            x, k = carry
            k, subkey = jr.split(k)
            p = jnp.where(self.dag[i] == -1, mu, pi[x[self.dag[i]]])
            x = x.at[i].set(jr.choice(subkey, pi.shape[0], p=p))
            return x, k

        x, _ = lax.fori_loop(0, len(self.dag), step, (x, seq_key))
        test_token = jr.choice(test_key, self.vocab_size)
        x = x.at[-1].set(test_token)
        y = pi[test_token]
        return x, y

    def bayes(self, seq, alpha=None):
        s, seq = seq[-1], seq[:-1]
        counts = jnp.zeros(self.vocab_size)
        counts = counts.at[seq].add(seq[self.dag] == s)
        counts += self.alpha
        return counts / counts.sum()

    def bigram(self, seq, eps=1e-8, alpha=None):
        s, seq = seq[-1], seq[:-1]
        counts = jnp.zeros(self.vocab_size)
        counts = counts.at[seq].add(seq[self.dag] == s)
        counts += eps
        return counts / counts.sum()

    def unigram(self, seq, alpha=None):
        counts = jnp.zeros(self.vocab_size)
        counts = counts.at[seq].add(1)
        counts += self.alpha
        return counts / counts.sum()


class InContextDAG:
    def __init__(self, vocab_size, dag, alpha):
        for i, p in enumerate(dag):
            # print(i, p)
            assert max(p, default=-1) < i
        dag = [jnp.array(p, dtype=int) for p in dag]
        self.vocab_size = vocab_size
        self.dag = dag
        self.alpha = alpha

    def sample(self, key):
        pi_key, seq_key = jr.split(key)
        ks = set(len(p) for p in self.dag)
        pi_keys = jr.split(pi_key, len(ks))
        pi = dict()
        pi[0] = jnp.ones(self.vocab_size) / self.vocab_size
        prior = self.alpha * jnp.ones(self.vocab_size)
        for k, subkey in zip(ks, pi_keys):
            pi[k] = jr.dirichlet(subkey, prior, [self.vocab_size] * k)

        x = jnp.zeros((len(self.dag) - 1,), dtype=int)
        for i in range(len(self.dag)):
            k = len(self.dag[i])
            if k == 0:
                p = pi[0]
            else:
                p = pi[k][tuple(x[self.dag[i]])]

            # if i != len(self.dag) - 1:
            #     seq_key, subkey = jr.split(seq_key)
            #     new_token = jr.choice(subkey, self.vocab_size, p=p)
            #     x = x.at[i].set(new_token)

            seq_key, subkey = jr.split(seq_key)
            new_token = jr.choice(subkey, self.vocab_size, p=p)
            x = x.at[i].set(new_token)

        return x, p

    def sample_whole(self, key):
        pi_key, seq_key = jr.split(key)
        ks = set(len(p) for p in self.dag)
        pi_keys = jr.split(pi_key, len(ks))
        pi = dict()
        pi[0] = jnp.ones(self.vocab_size) / self.vocab_size
        prior = self.alpha * jnp.ones(self.vocab_size)
        for k, subkey in zip(ks, pi_keys):
            pi[k] = jr.dirichlet(subkey, prior, [self.vocab_size] * k)

        x = jnp.zeros((len(self.dag),), dtype=int)
        for i in range(len(self.dag)):
            k = len(self.dag[i])
            if k == 0:
                p = pi[0]
            else:
                p = pi[k][tuple(x[self.dag[i]])]

            # if i != len(self.dag) - 1:
            #     seq_key, subkey = jr.split(seq_key)
            #     new_token = jr.choice(subkey, self.vocab_size, p=p)
            #     x = x.at[i].set(new_token)

            seq_key, subkey = jr.split(seq_key)
            new_token = jr.choice(subkey, self.vocab_size, p=p)
            x = x.at[i].set(new_token)

        return x[:-1], x[1:]

    def bayes_causal(self, seq, alpha=None):
        sequence_length = seq.shape[0]
        predictions = jnp.zeros((sequence_length, self.vocab_size))

        for t in range(sequence_length):
            counts = jnp.zeros(self.vocab_size)
            s = seq[self.dag[t + 1]]
            for i in range(t):
                if len(self.dag[i]) == len(s):
                    counts = counts.at[seq[i]].add(jnp.all(seq[self.dag[i]] == s))
            if alpha is not None:
                counts += alpha
            else:
                counts += self.alpha
            predictions = predictions.at[t, :].set(counts / counts.sum())

        return predictions

    def unigram_causal(self, seq, alpha=0):
        sequence_length = seq.shape[0]
        predictions = jnp.zeros((sequence_length, self.vocab_size))

        for t in range(sequence_length):
            counts = jnp.zeros(self.vocab_size)
            counts = counts.at[seq[: t + 1]].add(1)
            counts += alpha
            predictions = predictions.at[t, :].set(counts / counts.sum())

        return predictions

    def bayes(self, seq, alpha=None):
        counts = jnp.zeros(self.vocab_size)
        s = seq[self.dag[-1]]
        for i in range(len(self.dag) - 1):
            if len(self.dag[i]) == len(s):
                counts = counts.at[seq[i]].add(jnp.all(seq[self.dag[i]] == s))
        if alpha is not None:
            counts += alpha
        else:
            counts += self.alpha
        return counts / counts.sum()

    def unigram(self, seq, alpha=0):
        counts = jnp.zeros(self.vocab_size)
        counts = counts.at[seq].add(1)
        counts += alpha
        return counts / counts.sum()


class InContextRegression:
    def __init__(
        self,
        function_type="linear",
        sequence_length=128,
        noise=0,
        x_min=-10,
        x_max=10,
    ):
        self.function_type = function_type
        self.noise = noise
        self.x_min = x_min
        self.x_max = x_max
        self.sequence_length = sequence_length

    def function(self, x, fun_key):
        if self.function_type == "linear":
            w = jr.uniform(fun_key, (x.shape[0],))
            return jnp.dot(x, w)
        elif self.function_type == "quadratic":
            w = jr.uniform(fun_key, (x.shape[0],))
            return jnp.dot(x, w) + jnp.dot(x**2, w)
        elif self.function_type == "sin":
            w = jr.uniform(fun_key, (x.shape[0],))
            return jnp.dot(jnp.sin(x), w)
        elif self.function_type == "cos":
            w = jr.uniform(fun_key, (x.shape[0],))
            return jnp.dot(jnp.cos(x), w)
        else:
            raise ValueError(f"Unknown function type {self.function_type}")

    def sample(self, key):
        fun_key, seq_key = jr.split(key)

        x = jr.uniform(
            seq_key, (self.sequence_length,), minval=self.x_min, maxval=self.x_max
        )

        y = self.function(x, fun_key) + self.noise * jr.normal(
            fun_key, (self.sequence_length,)
        )

        sequence = jnp.empty((self.sequence_length * 2,))
        sequence = sequence.at[0::2].set(x)
        sequence = sequence.at[1::2].set(y)
        return sequence[:-1], sequence[-1]
