# Query Context-Aware Sequential Ranking

Provides implementations and evaluations of a query context-aware learning-to-rank model for sequential recommendation.

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

The notebooks in the `datasets` directory contain code to automatically download and preprocess the datasets.
After running these notebooks, the processed training and test data will be saved as Parquet files in the `files` directory.

## Training procedure

The entry point to start training the model is the `train_model.py`.

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


## Usage

Below are example commands for training the model with different integration types and options. Replace paths as needed.

**Quick experiment with a small subset:**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --num_epochs 2 --num_samples 10000 --batch_size 32
```

**No query context (NO_QUERY_CONTEXT):**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration NO_QUERY_CONTEXT
```

**Query context outside transformer blocks (OUTSIDE):**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration OUTSIDE
```

**Query context concatenated in input (IN_INPUT):**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT
```

**IN_INPUT with query context dropout (rate 0.25, enabled during training):**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --query_context_dropout_rate 0.25 --query_context_dropout_in_train 1
```

**IN_INPUT using historical query context during evaluation:**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration IN_INPUT --past_query_context_in_test 1
```

**Query context in last layer and outside transformer blocks (LAST_LAYER_AND_OUTSIDE):**
```bash
python train_model.py --dataset taobao --train_path datasets/files --test_path datasets/files --output_data_path ./model_output --integration LAST_LAYER_AND_OUTSIDE
```
