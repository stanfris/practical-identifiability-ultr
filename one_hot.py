import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from optax.losses import sigmoid_binary_cross_entropy


class TwoTowerModel(nnx.Module):
    def __init__(self, positions: int, query_doc_features: int, rngs: nnx.Rngs, bias_init_values: jnp.ndarray | None = None):
        self.positions = positions
        self.bias = nnx.Linear(
            in_features=positions,
            out_features=1,
            use_bias=False,
            rngs=rngs,
        )
        self.relevance = nnx.Linear(
            in_features=query_doc_features,
            out_features=1,
            use_bias=False,
            rngs=rngs,
        )
        if bias_init_values is not None:
            bias_init_values = bias_init_values.reshape((positions, 1))
            self.bias.kernel.value = bias_init_values

    def __call__(self, batch):
        bias = self.bias(batch["positions"])
        relevance = self.relevance(batch["query_doc_features"])
        return (bias + relevance).squeeze(-1)

    def loss_fn(self, batch):
        logits = self(batch)
        labels = batch["clicks"]
        return sigmoid_binary_cross_entropy(logits, labels).mean()

    def get_position_bias(self):
        positions_features = jax.nn.one_hot(jnp.arange(self.positions), self.positions)
        bias = self.bias(positions_features).squeeze()
        return bias - bias[0]


def train_model(model, batch, epochs, debug=False, freeze_bias_tower=False):

    optim = optax.adamw(0.005)

    if freeze_bias_tower:
        print("Freezing bias tower during training.")

        mask = freeze_bias_tower_mask(model)

        optim = optax.multi_transform(
            {
                "train": optim,
                "frozen": optax.set_to_zero(),
            },
            mask
        )
    else:   
        optim = optim

        
    optimizer = nnx.Optimizer(model, optim)
    loss = 0.0

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            return model.loss_fn(batch)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    for epoch in range(epochs):
        loss = train_step(model, optimizer, batch)

        if debug:
            print(f"Epoch {epoch}: loss = {loss:.4f}")

    return model, loss

def freeze_bias_tower_mask(model: nnx.Module):
        """
        Returns a pytree of labels ('train' or 'frozen') for optax.multi_transform.
        """
        # Get a structured dict of model state (params + other metadata)
        _, params, _ = nnx.split(model, nnx.Param, ...)

        def label_fn(path, _):
            # path is a tuple of keys, e.g. ("bias_tower", "embedding", "embedding")
            if any("bias" in str(k) for k in path):
                return "frozen"
            else:
                return "train"

        return jax.tree_util.tree_map_with_path(label_fn, params)
    
def test_model(model, batch):
    loss = model.loss_fn(batch)
    return loss

def prepare_batched_data(
    rngs,
    query_doc_features,
    query_doc_relevance,
    position_bias,
    positions_features,
    click_sessions,
    queries,
    positions,
    temperature,
):
    batch_query_doc_features = []
    batch_positions = []
    batch_clicks = []

    for _ in range(click_sessions):
        query_idx = jax.random.randint(rngs(), shape=(), minval=0, maxval=queries)

        if jax.random.uniform(rngs()) < temperature:
            sampled_position_idx = jax.random.permutation(rngs(), jnp.arange(positions))
        else:
            sampled_position_idx = jnp.arange(positions)

        sampled_query_doc_features = query_doc_features[query_idx, sampled_position_idx]
        sampled_query_doc_relevance = query_doc_relevance[
            query_idx, sampled_position_idx
        ]

        click_probs = nnx.sigmoid(sampled_query_doc_relevance + position_bias)
        clicks = jax.random.bernoulli(rngs(), click_probs).astype(jnp.float32)

        batch_positions.append(positions_features)
        batch_query_doc_features.append(sampled_query_doc_features)
        batch_clicks.append(clicks)

    batch = {
        "query_doc_features": jnp.stack(batch_query_doc_features),
        "positions": jnp.stack(batch_positions),
        "clicks": jnp.stack(batch_clicks),
    }

    return batch
    
if __name__ == "__main__":
    rngs = nnx.Rngs(42)

    queries = 1
    positions = 5
    query_doc_pairs = queries * positions
    click_sessions = 5_000
    temperature = 0.0
    fix_bias_tower = True

    # Generate one-hot encoded query-doc-features:
    query_doc_features = jax.nn.one_hot(jnp.arange(query_doc_pairs), query_doc_pairs)
    query_doc_features = query_doc_features.reshape(queries, positions, -1)

    # One-hot encoded position features:
    positions_features = jax.nn.one_hot(jnp.arange(positions), positions)

    # Sample random relevance scores for each query-doc pair:
    query_doc_relevance = jax.random.truncated_normal(
        rngs(), -2, 2, shape=(queries, positions)
    )
    
    # Ground-truth position bias
    position_bias = -np.log(np.arange(positions) + 1)

    print(f"Query-doc-features shape: {query_doc_features.shape}")
    print(f"True bias parameters:\n {position_bias.round(2)}")

    batch = prepare_batched_data(
        rngs,
        query_doc_features,
        query_doc_relevance,
        position_bias,
        positions_features,
        click_sessions,
        queries,
        positions,
        temperature,
    )

    batch_tmp_1 = prepare_batched_data(
        rngs,
        query_doc_features,
        query_doc_relevance,
        position_bias,
        positions_features,
        click_sessions,
        queries,
        positions,
        temperature=1.0,
    )

    # Train multiple models on the same dataset.
    # For simplicity in this notebook, if bias parameters vary widely across runs,
    # the model is not identifiable which should be always the case when temperature = 0,
    # no matter how many queries are being used.
    for run in range(5):
        model = TwoTowerModel(
            positions=positions,
            query_doc_features=query_doc_pairs,
            rngs=rngs,
        )

        model, loss = train_model(model, batch, epochs=1_000)
        estimated_bias = model.get_position_bias()
        print(
            f"Run {run} - estimated bias:",
            estimated_bias.round(2),
            "Final loss:",
            loss.round(2),
        )
        test_loss = test_model(model, batch_tmp_1)
        print(f"Run {run} - test loss at temperature=1.0: {test_loss.round(2)}")

if fix_bias_tower:
    print("\n--- Fixing bias tower during training ---\n")
    for run in range(5):
        # compute custom bias initialization
        position_bias = -np.log(np.arange(positions) + 1)
        position_bias[1] += run
        print("Initial bias values:", position_bias)

        # initialize model with custom bias tower weights
        model = TwoTowerModel(
            positions=positions,
            query_doc_features=query_doc_pairs,
            rngs=rngs,
            bias_init_values=position_bias,  
        )

        # train with bias tower frozen
        model, loss = train_model(
            model,
            batch,
            epochs=1_000,
            freeze_bias_tower=True,  
        )

        estimated_bias = model.get_position_bias()
        print(
            f"Run {run} - estimated bias:",
            estimated_bias.round(2),
            "Final loss:",
            loss.round(2),
        )

        test_loss = test_model(model, batch_tmp_1)
        print(f"Run {run} - test loss at temperature=1.0: {test_loss.round(2)}")
