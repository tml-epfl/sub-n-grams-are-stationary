from pathlib import Path

import optax
import tyro
from util import *
from PIL import Image

import wandb
from models import *
from plots import *
from problems import *


def main(
    vocab_size: int = 5,
    seq_len: int = 128,
    alpha: float = 1,
    dag: str = "markov",
    seed: int = 0,
    lr: float = 1e-2,
    wd: float = 1e-4,
    steps: int = 2**10,
    n_save: int = 2**10,
    batch_size: int = 2**7,
    max_size: int = 2**15,
    model_transformer: str = "transformer",
    optimizer: str = "adamw",
    scheduler: str = "cosine",
    mlp_trans: bool = True,
    log_out: bool = False,
    run_name: str = "test",
    save_plot_every: int = 2,
    momentum: float = 0,
    value_matrix: bool = True,
    qk: bool = True,
    lag: int = 0,
    custom_heads: int = 0,
    order: int = 1,
    skip: bool = True,
    dratio: float = 1.0,
    transem: str = "learned",
    scale: float = 1.0,
):

    config = locals()
    rng = RNG(seed)

    ### GENERATE DAG

    if "markov" in dag.lower():
        dag = generate_markov_dag(seq_len, lag, order)
        heads = [order, 1]
    else:
        raise NotImplementedError("Only 'markov' option is available.")

    if custom_heads != 0:
        heads = [custom_heads, 1]

    problem = InContextDAG(
        vocab_size=vocab_size,
        dag=dag,
        alpha=alpha,
    )
    dag = problem.dag

    #### MODELS
    common_params = {
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "heads": heads,
        "key": rng.next(),
        "qk": qk,
        "use_log": log_out,
        "d_QK_ratio": dratio,
        "scale": scale,
    }

    model_mapping = {
        "catformer": CatFormer,
        "transformer": Transformer,
        "simple": Simpleformer,
    }

    if model_transformer in model_mapping:
        model_class = model_mapping[model_transformer]
        if model_transformer == "transformer":
            transformer_specific_params = {
                "embedding_type": transem,  # or "onehot" or "learned"
                "use_skip": skip,
                "use_mlp": mlp_trans,
            }
            model = model_class(**common_params, **transformer_specific_params)
        elif model_transformer == "simple":
            simple_specific_params = {
                "value": value_matrix,
            }
            model = model_class(**common_params, **simple_specific_params)
        else:
            model = model_class(**common_params)

    ### Training Set-up
    @jit
    def criterion(f, y):
        _criterion = lambda f, y: -jnp.log(f + 1e-8) @ y
        for _ in range(y.ndim - 1):
            _criterion = vmap(_criterion)
        return _criterion(f, y).mean()

    @jit
    def loss_fn(model, batch):
        x, y = batch
        return criterion(model(x), y)

    A = jnp.zeros((seq_len + 1, seq_len), dtype=int)
    for i in range(seq_len + 1):
        A = A.at[i, dag[i]].set(1)

    ########## Inspection for optimal constatnt to add for lower order estimators for baseline
    def compute_lower_order_estimators(
        dag, order, lag, alpha_values, testx, testy, vocab_size, alpha
    ):
        def create_bigram_dag(dag, order, lag, offset):
            new_dag = dag[: order + lag] + [
                [x[-offset - lag]] for x in dag[order + lag :]
            ]
            return InContextDAG(vocab_size=vocab_size, dag=new_dag, alpha=alpha)

        def compute_min_criterion(dag, alpha_values, testx, testy):
            values = []
            for alpha_t in alpha_values:
                logits = vmap(dag.bayes, in_axes=(0, None))(testx, alpha_t)
                criterion_value = criterion(logits, testy)
                values.append(criterion_value)
            return min(values)

        bigram_1 = None
        bigram_2 = None
        bigram_3 = None
        trigram = None

        if order >= 1:
            unigram_values = []
            for alpha_t in alpha_values:
                l_unigram = vmap(problem.unigram, in_axes=(0, None))(testx, alpha_t)
                unigram = criterion(l_unigram, testy)
                unigram_values.append(unigram)
            unigram = min(unigram_values)

        if order >= 2:

            one_nn = create_bigram_dag(dag, order, lag, 1)
            bigram_1 = compute_min_criterion(one_nn, alpha_values, testx, testy)

            one_nn_2 = create_bigram_dag(dag, order, lag, 2)
            bigram_2 = compute_min_criterion(one_nn_2, alpha_values, testx, testy)

        if order >= 3:
            one_nn_3 = create_bigram_dag(dag, order, lag, 3)
            bigram_3 = compute_min_criterion(one_nn_3, alpha_values, testx, testy)

            two_nn_dag = dag[: order + lag] + [
                [x[-2 - lag], x[-1 - lag]] for x in dag[order + lag :]
            ]
            two_nn = InContextDAG(vocab_size=vocab_size, dag=two_nn_dag, alpha=alpha)
            trigram = compute_min_criterion(two_nn, alpha_values, testx, testy)

        return unigram, bigram_1, bigram_2, bigram_3, trigram

    print("Computing Bayes")
    testx, testy = vmap(problem.sample)(rng.next(2**14))
    logits = vmap(problem.bayes)(testx)
    bayes = criterion(logits, testy)

    print("Computing lower order estimators")
    # alpha_values = alpha * jnp.logspace(-1, 2, 100)
    alpha_values = [alpha]

    unigram, bigram_1, bigram_2, bigram_3, trigram = compute_lower_order_estimators(
        dag, order, lag, alpha_values, testx, testy, vocab_size, alpha
    )

    #############

    print("Training")
    save_every = steps // n_save
    epoch_len = max_size // batch_size
    sample_fn = jit(lambda k: vmap(problem.sample)(jr.split(k, epoch_len * batch_size)))

    def batch_iterator(key):
        while True:
            key, subkey = jr.split(key)
            batches = sample_fn(subkey)
            for i in range(epoch_len):
                yield tree_map(
                    lambda x: x[batch_size * i : batch_size * (i + 1)], batches
                )

    if scheduler == "cosine":
        schedule = optax.cosine_decay_schedule(lr, steps)
    else:
        schedule = optax.constant_schedule(lr)

    if optimizer == "adamw":
        opt = optax.adamw(schedule, weight_decay=wd, b1=momentum)
    elif optimizer == "sgd":
        opt = optax.sgd(schedule, momentum=momentum)

    @jit
    def step_fn(model, batch, opt_state):
        g = jax.grad(loss_fn)(model, batch)
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(g))
        )
        for p in jax.tree_util.tree_leaves(model):
            print(p)
            print(jnp.sum(jnp.square(p)))

        param_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(model))
        )
        print(param_norm)
        scaled_grad_norm = grad_norm / param_norm
        updates, opt_state = opt.update(g, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, grad_norm, scaled_grad_norm

    iterator = batch_iterator(rng.next())

    opt_state = opt.init(model)

    test_losses = []
    pbar = tqdm(total=steps)
    wandb.init(project="sub-n-grams", config=config, name=run_name)
    grad_norm = None
    scaled_grad_norm = None
    for i in range(steps):
        if i % save_every == 0:
            test_loss = loss_fn(model, (testx, testy))
            test_losses.append(test_loss)
            wandb.log(
                dict(
                    loss=test_loss,
                    bayes=bayes,
                    step=i,
                    lr=schedule(i),
                    unigram=unigram,
                    bigram=bigram_1,
                    bigram2=bigram_2,
                    bigram3=bigram_3,
                    trigram=trigram,
                    loss_excess=test_loss - bayes,
                    grad_norm=gns / gns_i,
                    scaled_grad_norm=scaled_grad_norm,
                ),
                step=i,
            )

            norms = get_parameter_norms_dict(model)
            wandb.log(norms, step=i)

            pbar.n = i

            if int(i / save_every) % save_plot_every == 0:
                if model_transformer == "simple":
                    plot_model_simple(model, qk=qk, step=i)
                elif model_transformer == "transformer":
                    plot_model_transformer(model, qk=qk, input=testx[0, :], step=i)
                elif model_transformer == "catformer":
                    plot_model_catformer(
                        model,
                        qk=qk,
                        seq_len=seq_len,
                        vocab_size=vocab_size,
                        input=testx[0, :],
                        step=i,
                    )

            gns = 0
            gns_i = 0

        model, opt_state, grad_norm, scaled_grad_norm = step_fn(
            model, next(iterator), opt_state
        )
        gns += grad_norm
        gns_i += 1

    pbar.n = steps
    pbar.refresh()
    pbar.close()
    test_losses = jnp.array(test_losses)

    # Final Losses Plot
    fig = plot_losses(
        test_losses,
        bayes,
        save_every,
        bigram=bigram_1,
        unigram=unigram,
        trigram=trigram,
    )
    filename = wandb.run.dir + "/losses.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({"losses/test": wandb.Image(Image.open(filename))})
    plt.close(fig)

    # A(Causal Graph) Plot
    fig = plot_A(A)
    filename = wandb.run.dir + "/A.png"
    plt.savefig(filename, bbox_inches="tight", dpi=100)
    wandb.log({"A": wandb.Image(Image.open(filename))})
    plt.close(fig)

    wandb.finish()


tyro.cli(main)
