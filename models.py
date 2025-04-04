import jax
from jax import nn
from jax import numpy as jnp
from jax import random as jr
from simple_pytree import Pytree, static_field


class muLinear(Pytree):
    def __init__(self, input_dim, output_dim, key, bias=True, zero_init=False):
        if zero_init:
            self.W = jnp.zeros([input_dim, output_dim])
        else:
            self.W = jr.normal(key, [input_dim, output_dim]) / jnp.sqrt(output_dim)
        if bias:
            self.b = jnp.zeros(output_dim)

    def __call__(self, x):
        x @= self.W * jnp.sqrt(self.W.shape[1] / self.W.shape[0])
        if hasattr(self, "b"):
            x += self.b * jnp.sqrt(len(self.b))
        return x


class MLP(Pytree):
    activation = static_field()

    def __init__(self, widths, activation, key):
        self.activation = activation

        keys = jr.split(key, len(widths) - 1)
        layers = []
        for i in range(len(widths) - 2):
            layers.append(muLinear(widths[i], widths[i + 1], keys[i]))
        layers.append(muLinear(widths[-2], widths[-1], keys[-1], zero_init=True))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


class Simpleformer(Pytree):
    vocab_size: int = static_field()
    QK: bool = static_field()
    value: bool = static_field()
    use_mlp: bool = static_field(default=False)
    use_log: bool = static_field(default=False)

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def attn2(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x[:, :, : self.vocab_size])
        return attn

    def attn1(self, x, A):
        T = A.shape[-1]
        attn = jnp.where(jnp.tri(T), A, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        wte = jnp.eye(self.vocab_size)[x]
        return wte

    def __init__(
        self,
        seq_len,
        vocab_size,
        heads,
        key,
        value=True,
        qk=True,
        d_QK_ratio=1,
        use_mlp=False,
        use_log=False,
        scale=1.0,
    ):
        self.vocab_size = vocab_size
        self.QK = qk
        self.value = value
        self.use_mlp = use_mlp
        self.use_log = use_log

        d = vocab_size
        keys = jr.split(key, 4 * len(heads) + 4)

        self.A = []
        self.V = []
        for layer, n_head in enumerate(heads):
            if layer == 0:
                self.A.append(jnp.zeros([n_head, seq_len, seq_len]))
                self.V.append(jr.normal(keys[1], [n_head, d, d]) / jnp.sqrt(d))
                d *= 1 + n_head
            else:
                d_hidden_QK = int(d * d_QK_ratio)
                if qk:
                    self.Q = (
                        scale
                        * jr.normal(keys[2], [n_head, d, d_hidden_QK])
                        / jnp.sqrt(d)
                    )
                    self.Kt = (
                        scale
                        * jr.normal(keys[4], [n_head, d_hidden_QK, d])
                        / jnp.sqrt(d)
                    )
                    self.A.append([])
                else:
                    self.A.append(jnp.zeros([n_head, d, d]))

        if self.use_mlp:
            self.mlp = MLP(
                widths=[vocab_size, 8 * vocab_size, vocab_size],
                activation=nn.relu,
                key=keys[0],
            )

    def __call__(self, x):
        x = self.embed(x)
        for layer, Ai in enumerate(self.A):
            if layer == 0:
                Vi = self.V[layer]
                attn = jax.vmap(self.attn1, (None, 0), -2)(x, Ai)
                if self.value:
                    attn = jnp.einsum("...ijk,jkl->...ijl", attn, Vi)
                attn = attn.reshape(*attn.shape[:-2], -1)
                x = jnp.concatenate([x, attn], -1)
            elif layer == 1:
                if self.QK:
                    attn = jax.vmap(self.attn2, (None, 0), -2)(
                        x, jnp.matmul(self.Q, self.Kt)
                    )
                else:
                    attn = jax.vmap(self.attn2, (None, 0), -2)(x, Ai)
                x = attn.reshape(*attn.shape[:-2], -1)
        x = x[..., -1, :]

        if self.use_mlp:
            x = jax.vmap(self.mlp)(x)

        if self.use_log:
            x = jnp.log(jnp.abs(x) + 1e-8)

        return x


class CatFormer(Pytree):
    vocab_size: int = static_field()
    use_mlp: bool = static_field(default=False)
    use_log: bool = static_field(default=False)
    qk: bool = static_field(default=False)

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        wte = jnp.eye(self.vocab_size)[x]
        wpe = jnp.eye(x.shape[-1])
        wpe = jnp.broadcast_to(wpe, (*x.shape, x.shape[-1]))
        return jnp.concatenate([wte, wpe], -1)

    def __init__(
        self,
        seq_len,
        vocab_size,
        heads,
        key=None,
        use_mlp=False,
        use_log=False,
        qk=False,
        d_QK_ratio=1,
        scale=1.0,
    ):
        self.vocab_size = vocab_size
        self.use_mlp = use_mlp
        self.use_log = use_log
        self.qk = qk

        d = seq_len + vocab_size
        d_hidden = int(d * d_QK_ratio)
        self.A = []
        self.Q = []
        self.Kt = []
        self.V = []
        self.mlps = []

        if qk:
            keys = jr.split(key, 2 * len(heads) + 2)
            for i, n_head in enumerate(heads):
                self.Q.append(
                    scale * jr.normal(keys[2 * i], [n_head, d, d_hidden]) / jnp.sqrt(d)
                )
                self.Kt.append(
                    scale
                    * jr.normal(keys[2 * i + 1], [n_head, d_hidden, d])
                    / jnp.sqrt(d)
                )
                d *= 1 + n_head
            if use_mlp:
                self.mlp = MLP(
                    widths=[vocab_size, 8 * vocab_size, vocab_size],
                    activation=nn.relu,
                    key=keys[0],
                )
        else:
            for n_head in heads:
                self.A.append(jnp.zeros([n_head, d, d]))
                d *= 1 + n_head

        self.W = jnp.zeros((d, vocab_size))

    def __call__(self, x):
        x = self.embed(x)
        if self.qk:
            for i in range(len(self.Q)):
                Ai = jnp.matmul(self.Q[i], self.Kt[i])
                attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
                attn = attn.reshape(*attn.shape[:-2], -1)
                x = jnp.concatenate([x, attn], -1)
        else:
            for Ai in self.A:
                attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
                attn = attn.reshape(*attn.shape[:-2], -1)
                x = jnp.concatenate([x, attn], -1)
        x = x[..., -1, :]
        x = x @ self.W

        if self.use_mlp:
            x = jax.vmap(self.mlp)(x)

        if self.use_log:
            x = jnp.log(jnp.abs(x) + 1e-8)

        return nn.softmax(x)


class Transformer(Pytree):
    vocab_size: int = static_field()
    use_mlp: bool = static_field(default=False)
    use_log: bool = static_field(default=False)
    qk: bool = static_field(default=False)
    use_skip: bool = static_field(default=True)
    embedding_type: str = static_field(default="learned")

    def attn(self, x, A):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jk,...lk -> ...il", x, A, x)
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn

    def embed(self, x):
        if self.embedding_type == "learned":
            wte = self.wte[x]
            wpe = jnp.broadcast_to(self.wpe, (*x.shape, self.wpe.shape[-1]))
            return wte + wpe
        elif self.embedding_type == "onehot":
            wte = jnp.eye(self.vocab_size)[x]
            wpe = jnp.eye(x.shape[-1])
            wpe = jnp.broadcast_to(wpe, (*x.shape, x.shape[-1]))
            return jnp.concatenate([wte, wpe], -1)

    def __init__(
        self,
        seq_len,
        vocab_size,
        heads,
        key,
        use_mlp=False,
        use_log=False,
        qk=True,
        d_QK_ratio=1,
        use_skip=True,
        embedding_type="learned",
        scale=1.0,
    ):
        self.vocab_size = vocab_size
        self.use_mlp = use_mlp
        self.use_log = use_log
        self.qk = qk
        d = seq_len + vocab_size
        d_hidden = int(d * d_QK_ratio)
        self.A = []
        self.Q = []
        self.Kt = []
        self.V = []
        self.mlps = []
        self.use_skip = use_skip
        self.embedding_type = embedding_type

        keys = jr.split(key, 4 * len(heads) + 4 if qk else 2 * len(heads) + 2)
        for i in range(len(heads)):
            n_head = heads[i]
            if qk:
                self.Q.append(
                    scale * jr.normal(keys[4 * i], [n_head, d, d_hidden]) / jnp.sqrt(d)
                )
                self.Kt.append(
                    scale
                    * jr.normal(keys[4 * i + 1], [n_head, d_hidden, d])
                    / jnp.sqrt(d)
                )
                self.V.append(
                    scale * jr.normal(keys[4 * i + 2], [n_head, d, d]) / jnp.sqrt(d)
                )
            else:
                self.A.append(jnp.zeros([n_head, d, d]))
                self.V.append(jr.normal(keys[2 * i], [n_head, d, d]) / jnp.sqrt(d))

            if use_mlp:
                self.mlps.append(
                    MLP(widths=[d, d, d], activation=nn.relu, key=keys[-3])

                )
        self.W = jr.normal( keys[0], [d, vocab_size]) / jnp.sqrt(d)

        self.wte = jr.normal(keys[-2], [vocab_size, d]) / jnp.sqrt(d)
        self.wpe = jr.normal(keys[-1], [seq_len, d]) / jnp.sqrt(d)

    def get_attn(self, x):
        x = self.embed(x)

    def __call__(self, x):
        x = self.embed(x)

        if self.qk:
            for i in range(len(self.Q)):
                Ai = jnp.matmul(self.Q[i], self.Kt[i])
                Vi = self.V[i]
                if self.use_mlp:
                    mlpi = self.mlps[i]
                attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
                attn = jnp.einsum("...ijk,jkl->...ijl", attn, Vi)
                delta = jnp.sum(attn, axis=-2)
                if self.use_mlp:
                    delta = jax.vmap(mlpi)(delta)
                if self.use_skip:
                    x = x + delta
                else:
                    x = delta
        else:
            for i in range(len(self.A)):
                Ai = self.A[i]
                Vi = self.V[i]
                if self.use_mlp:
                    mlpi = self.mlps[i]
                attn = jax.vmap(self.attn, (None, 0), -2)(x, Ai)
                attn = jnp.einsum("...ijk,jkl->...ijl", attn, Vi)
                delta = jnp.sum(attn, axis=-2)
                if self.use_mlp:
                    delta = jax.vmap(mlpi)(delta)
                if self.use_skip:
                    x = x + delta
                else:
                    x = delta

        x = x[..., -1, :]
        z = x @ self.W

        if self.use_log:
            z = jnp.log(jnp.abs(z) + 1e-8)

        return nn.softmax(z)
