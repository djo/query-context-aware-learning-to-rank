# Query Context-Aware Sequential Ranking

This repository provides the implementation and evaluation of the methods described in the paper
Efficient and Effective Query Context-Aware Learning-to-Rank Model for Sequential Recommendation.

## Folder structure

```
├── datasets/                # Contains preprocessing notebooks
│   ├── {preprocessing notebooks for Taobao and RetailRocket}
├── files/                   # Stores processed dataset files
├── model.py                 # Model definition
├── train_model.py           # Script for model training
└── README.md                # Documentation
```
## Data

Notebooks in `datasets` subdirectory provide code to download and preprocess the datasets.
Their execution results in `files` directory being populated with the respective test and train parquet files.

## Training procedure

The entry point to start training the model is the `train_model.py` script.

Parameters:

* `--dataset`: either `taobao` or `retailrocket`, to select the respective dataset
* `--train_path`: path to the local folder, storing the training datasets
* `--test_path`: path to the local folder, storing the training datasets
* `--output_data_path`: path for saving the resulting model artifact
* `--integration`: one of the following options (default: NO_QUERY_CONTEXT):
  - `NO_QUERY_CONTEXT` for no query context (the respective input will be ignored, and only the items sequence will be used)
  - `OUTSIDE` for query context outside the transformer blocks
  - `IN_INPUT` for implementation where query context is concatenated with the preceding item representation in the input
  - `LAST_LAYER_AND_OUTSIDE` for query context included in the last layer's query position and outside the transformer blocks
* `--num_epochs`: number of training epochs (default: 30)
* `--num_samples`: number of samples to use from dataset; use -1 for all samples (default: -1)
* `--seq_len`: maximum sequence length for input data (default: 100)
* `--batch_size`: batch size for training (default: 128)
* `--past_query_context_in_test`: whether to use past query context information during evaluation (0=only the current context is used, 1=historical context is used as well, default: 0)
* `--query_context_dropout_rate`: dropout rate for query context information; applicable only for `IN_INPUT` integration type (default: 0.0)
* `--query_context_dropout_in_train`: whether to apply query context dropout during training; applicable only for `IN_INPUT` integration typ (0=disabled, 1=enabled, default: 0)

## Usage example

To experiment locally with a small subset of data (10K):

To debug locally, one can use the following command

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --num_epochs 2 --num_samples 10_000 --batch_size 32
```

All the experiments were conducted with the default hyperparameters from the repo

## Executions used in the paper evaluation

All hyperparameters used are set as defaults in the project and can be found in the source code configuration.
Therefore, only the integration type needs to be specified.

NO_QUERY_CONTEXT integration type:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration NO_QUERY_CONTEXT
```

OUTSIDE integration type:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration OUTSIDE
```

IN_INPUT integration type:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT
```

IN_INPUT integration type with the dropout rate 0.25:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --query_context_dropout_rate 0.25 --query_context_dropout_in_train 1
```

IN_INPUT integration type with the historical query context in evaluation:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --past_query_context_in_test 1 
```

LAST_LAYER_AND_OUTSIDE integration type:

```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration LAST_LAYER_AND_OUTSIDE
```
