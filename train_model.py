import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import argparse

import pyarrow.parquet as pq
import numpy as np

import tensorflow as tf
import tensorflow_ranking as tfr

from model import Model
from model import Config
from model import QueryContextIntegration


DATASETS = {
    "taobao": {
        "path_train": "train_taobao_preprocessed.parquet",
        "path_test": "test_taobao_preprocessed.parquet",
        "num_items": 315689,
        "num_queries": 7905,
    },
    "retailrocket": {
        "path_train": "train_retailrocket_preprocessed.parquet",
        "path_test": "test_retailrocket_preprocessed.parquet",
        "num_items": 46893,
        "num_queries": 1006,
    },
}

SHUFFLE_BUFFER_SIZE = 10_000


def read_parquet_dataset(file_path, seq_len=100):
    """
    Read raw data and create a tf.data.Dataset,
    where each data point is represented as a sequence of items with their attributes.
    The sequence is left-padded, shifting all items and attributes to the right.
    """

    def _read_row(row):
        return {
            k: np.pad(row[k][-seq_len:], (seq_len - len(row[k][-seq_len:]), 0), constant_values=(0, 0)).astype(np.int32)
            for k in row.keys()
            if k not in ["user_id"]
        }

    parquet_file = pq.ParquetFile(file_path)

    def generator():
        for record_batch in parquet_file.iter_batches():
            df = record_batch.to_pandas()
            for index, row in df.iterrows():
                yield _read_row(row)

    return tf.data.Dataset.from_generator(
        generator,
        output_signature={
            "items": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            "categories": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            "timestamps": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
            "behaviors": tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        },
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=list(DATASETS.keys()), default="taobao")
    parser.add_argument("--train_path", default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test_path", default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--output_data_path", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument(
        "--integration", type=str, choices=QueryContextIntegration.__members__, default="NO_QUERY_CONTEXT"
    )
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--past_query_context_in_test", type=int, choices=[0, 1], default=0)
    parser.add_argument("--query_context_dropout_rate", type=float, default=0.0)
    parser.add_argument("--query_context_dropout_in_train", type=int, choices=[0, 1], default=0)

    args, unknown = parser.parse_known_args()
    print(f"args: {args}")
    print(f"unknown args: {unknown}")

    return args


if __name__ == "__main__":
    args = get_args()
    dataset = DATASETS[args.dataset]
    print(f"\n{args.dataset}: {dataset}")

    config = Config(
        num_items=dataset["num_items"] + 1,
        num_queries=dataset["num_queries"] + 1,
        num_attention_heads=2,
        num_layer_blocks=2,
        model_dim=128,
        max_len=args.seq_len,
        dropout_rate=0.1,
        integration=QueryContextIntegration[args.integration],
        past_query_context_in_test=bool(args.past_query_context_in_test),
        query_context_dropout_rate=args.query_context_dropout_rate,
        query_context_dropout_in_train=bool(args.query_context_dropout_in_train),
    )
    print(f"\n{config}\n")

    metrics = [
        tfr.keras.metrics.NDCGMetric(topn=5, name="ndcg_at_5"),
        tfr.keras.metrics.RecallMetric(topn=5, name="recall_at_5"),
        tfr.keras.metrics.NDCGMetric(topn=20, name="ndcg_at_20"),
        tfr.keras.metrics.RecallMetric(topn=20, name="recall_at_20"),
        tfr.keras.metrics.NDCGMetric(topn=50, name="ndcg_at_50"),
        tfr.keras.metrics.RecallMetric(topn=50, name="recall_at_50"),
    ]

    model = Model(config)

    train_data_path = os.path.join(args.train_path, dataset["path_train"])
    train_dataset = (
        read_parquet_dataset(train_data_path, seq_len=args.seq_len)
        .take(args.num_samples)
        .batch(args.batch_size)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_data_path = os.path.join(args.test_path, dataset["path_test"])
    test_dataset = (
        read_parquet_dataset(test_data_path, seq_len=args.seq_len)
        .take(args.num_samples)
        .batch(args.batch_size)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=metrics)
    model.fit(train_dataset, epochs=args.num_epochs, validation_data=test_dataset, validation_freq=1, verbose=2)
    model.summary(expand_nested=True, show_trainable=True)

    model.save_weights(os.path.join(args.output_data_path, "model_checkpoint"))
