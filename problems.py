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
