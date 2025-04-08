from enum import Enum
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import MultiHeadAttention


class QueryContextIntegration(Enum):
    NO_QUERY_CONTEXT = "no_query_context"
    OUTSIDE = "query_context_outside"
    IN_INPUT = "query_context_in_input"
    LAST_LAYER_AND_OUTSIDE = "query_context_last_layer_and_outside"


@dataclass
class Config:
    num_items: int
    num_queries: int
    num_attention_heads: int
    num_layer_blocks: int
    model_dim: int
    max_len: int
    dropout_rate: float
    integration: QueryContextIntegration
    past_query_context_in_test: bool
    query_context_dropout_rate: float
    query_context_dropout_in_train: bool


PADDING_IDX = 0


class FFNN(tf.keras.layers.Layer):
    def __init__(self, model_dim: int, dropout_rate: float, **kwargs):
        super().__init__(**kwargs)
        self.hidden_layer = tf.keras.layers.Dense(model_dim, activation="relu")
        self.output_layer = tf.keras.layers.Dense(model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs):
        return self.output_layer(self.dropout(self.hidden_layer(inputs)))


class LayerBlock(tf.keras.layers.Layer):
    """
    Residual attention-based layer block with the pre-norm implementation.
    """

    def __init__(self, name: str, config: Config, **kwargs):
        super().__init__(name=name, **kwargs)

        self.mha = MultiHeadAttention(
            num_heads=config.num_attention_heads,
            key_dim=config.model_dim,
            dropout=config.dropout_rate,
        )

        self.ffnn = FFNN(model_dim=config.model_dim, dropout_rate=config.dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.layernorm3 = tf.keras.layers.LayerNormalization()

        self.dropout1 = tf.keras.layers.Dropout(config.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(config.dropout_rate)

    def call(self, query, value, attention_mask):
        residual = value
        query = self.layernorm1(query)
        value = self.layernorm2(value)
        x = self.mha(query=query, value=value, attention_mask=attention_mask, use_causal_mask=True)
        x = residual + self.dropout1(x)

        residual = x
        x = self.ffnn(self.layernorm3(x))
        x = residual + self.dropout2(x)

        return x


class Model(tf.keras.Model):
    """
    Deep self-attention transformer model trained with causal language modeling.
    """

    def __init__(self, config: Config, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Embeddings
        self.item_embedding = tf.keras.layers.Embedding(
            input_dim=config.num_items + 1, output_dim=config.model_dim, name="item_embedding"
        )
        self.category_embedding = tf.keras.layers.Embedding(
            input_dim=config.num_queries + 1, output_dim=config.model_dim, name="category_embedding"
        )
        self.pos_embedding = tf.keras.layers.Embedding(
            input_dim=config.max_len + 1, output_dim=config.model_dim, mask_zero=True, name="pos_embedding"
        )

        # Additional projections for each query context integration type
        self.input_query_context_dense = tf.keras.layers.Dense(
            units=config.model_dim, activation="relu", name="input_query_context_dense"
        )
        self.last_layer_query_context_dense = tf.keras.layers.Dense(
            units=config.model_dim, activation="relu", name="last_layer_query_context_dense"
        )
        self.outside_ffnn = FFNN(model_dim=config.model_dim, dropout_rate=config.dropout_rate, name="outside_ffnn")

        # Projection for the item's contextual information case
        self.input_item_context_dense = tf.keras.layers.Dense(
            units=config.model_dim, activation="relu", name="input_item_context_dense"
        )

        # Transformer's residual self-attention layer blocks
        self.layer_blocks = [LayerBlock(name=f"layer_block_{i}", config=config) for i in range(config.num_layer_blocks)]
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        batch_size = tf.shape(inputs["items"])[0]
        seq_len = tf.shape(inputs["items"])[1]

        # drop the last item from the sequence and prepend padding (cold start case),
        # making the sequence consist of shifted past actions and input["items"] becomes their next-item targets
        shifted_sequence = _shift_right(inputs["items"])
        x = self.item_embedding(shifted_sequence)

        # apply the same shift same to the category sequence to obtain the item's context
        shifted_categories = _shift_right(inputs["categories"])
        item_context = self.category_embedding(shifted_categories)

        # next-item category becomes the query context
        query_context = self.category_embedding(inputs["categories"])

        if self.config.integration == QueryContextIntegration.IN_INPUT:
            print("Integration: query context is in the input")
            concatenated = tf.concat([x, query_context], axis=-1)
            x = self.input_query_context_dense(concatenated)

        # learnable positional encoding, see https://arxiv.org/abs/2405.10436 (Section 3.1)
        positions = tf.range(seq_len)
        positions = tf.expand_dims(positions, 0)  # [1, seq_len]
        positions = tf.tile(positions, [batch_size, 1])  # [batch_size, seq_len]
        x += self.pos_embedding(positions)

        padding_mask = tf.not_equal(inputs["items"], PADDING_IDX)
        attention_mask = tf.repeat(tf.expand_dims(padding_mask, 1), seq_len, axis=1)

        last_layer_block_index = len(self.layer_blocks) - 1

        # residual attention-based layer blocks
        for i, layer_block in enumerate(self.layer_blocks):
            query = x
            if (
                self.config.integration == QueryContextIntegration.LAST_LAYER_AND_OUTSIDE
                and i == last_layer_block_index
            ):
                print(f"Integration: query context is in the last layer '{layer_block.name}' query position")
                query = self.last_layer_query_context_dense(tf.concat([x, query_context], axis=-1))
            x = layer_block(query=query, value=x, attention_mask=attention_mask)

        # pre-norm requires a final normalization, see https://arxiv.org/pdf/1910.05895.pdf (Section 2.1)
        x = self.layernorm(x)

        if self.config.integration in (
            [QueryContextIntegration.OUTSIDE, QueryContextIntegration.LAST_LAYER_AND_OUTSIDE]
        ):
            print("Integration: query context is outside")
            x = self.outside_ffnn(tf.concat([x, query_context], axis=-1))

        return x

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            if self.config.query_context_dropout_in_train:
                print(f"Training: category dropout is applied with the rate {self.config.query_context_dropout_rate}")
                inputs = _with_category_dropout(inputs, self.config.query_context_dropout_rate)

            y_pred = self(inputs)
            padding_mask = tf.reshape(tf.not_equal(inputs["items"], PADDING_IDX), [-1])

            target = tf.boolean_mask(tf.reshape(inputs["items"], (-1, 1)), padding_mask)
            inputs = tf.boolean_mask(tf.reshape(y_pred, (-1, self.config.model_dim)), padding_mask)

            loss = tf.nn.sampled_softmax_loss(
                weights=self.item_embedding.weights[0],
                biases=tf.zeros(self.item_embedding.weights[0].shape[0]),
                labels=target,
                inputs=inputs,
                num_sampled=int(self.config.num_items * 0.005),
                remove_accidental_hits=True,
                num_classes=self.config.num_items,
            )

            loss = tf.reduce_mean(loss)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, inputs):
        if not self.config.past_query_context_in_test:
            print("Evaluation: padding is applied to the historical query context")
            inputs = _with_past_categories_padded(inputs)

        y_pred = self(inputs)
        target = tf.expand_dims(inputs["items"][:, -1], -1)

        per_item_scores = tf.matmul(y_pred[:, -1], self.item_embedding.weights[0], transpose_b=True)
        top_100_scores, top_100_items = tf.nn.top_k(per_item_scores, k=100)

        true_labels = tf.zeros_like(top_100_scores)
        target = tf.where(tf.equal(top_100_items, target), tf.ones_like(true_labels), true_labels)
        self.compiled_metrics.update_state(target, top_100_scores)

        return {m.name: m.result() for m in self.metrics}


def _shift_right(sequence):
    """
    Drops the last token from the sequence and prepends padding.
    """
    batch_size = tf.shape(sequence)[0]
    padding = tf.fill(dims=[batch_size, 1], value=PADDING_IDX)
    shifted_sequence = tf.concat([padding, sequence[:, :-1]], axis=1)
    return shifted_sequence


def _with_category_dropout(inputs, dropout_rate):
    """
    Randomly drops the category at each position to partially address the online-offline mismatch
    in the case of approach with the query context in the input.
    The mismatch refers to the scenario where historical context is unavailable during inference (online).
    """
    categories = inputs["categories"]
    batch_size = tf.shape(categories)[0]
    seq_len = tf.shape(categories)[1]
    # binomial distribution with the probability of success {1} as 1.0 - dropout_rate
    probs = tf.tile([[dropout_rate, 1.0 - dropout_rate]], multiples=[batch_size, 1])
    mask = tf.random.categorical(tf.math.log(probs), seq_len, dtype="int32")
    padding = tf.fill(dims=[batch_size, seq_len], value=PADDING_IDX)
    categories = tf.where(tf.equal(mask, 1), categories, padding)
    return {"items": inputs["items"], "categories": categories}


def _with_past_categories_padded(inputs):
    """
    Retains only the last category (query context) in the sequence while applying padding to past categories.
    The last item/query is used as the only target in the leave-one-out evaluation.
    This emulates a scenario where historical context is unavailable during inference (online).
    """
    padding = tf.fill(
        value=PADDING_IDX, dims=[tf.shape(inputs["categories"])[0], tf.shape(inputs["categories"])[1] - 1]
    )
    categories = tf.concat([padding, inputs["categories"][:, -1:]], axis=1)
    return {"items": inputs["items"], "categories": categories}
