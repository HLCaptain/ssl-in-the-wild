<div align="center">

# SSL in the Wild: comparison of Self-Supervised Learning methods on ImageNet

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

Homework project for the Deep Learning course at Budapest University of Technology and Economics.

Team:

- Balázs Püspök-Kiss (BL6ADS)
- Hanna Czifrus (H2MJ2U)
- Réka Farkas (IXAGK7)

## Project

### Self-supervised learning "in-the-wild"

Most of the popular self-supervised learning (SSL) methods are developed for pretraining on ImageNet and assessed on ImageNet-1k classification. Therefore, these methods are tuned and designed (i.e., “overfitted”) for this dataset and might perform worse on other datasets and data “from-the-wild". Testing this hypothesis on multiple datasets could provide valuable insights on the limitations of SSL, giving you hands-on experience with these methods. Task of the students: Explore open-source implementations of self-supervised learning methods, select at least two methods (with matching ImageNet-pretrained models available for each). Get familiar with the selected algorithms, hyperparameters, applied optimizers, etc. Select 1-2 classification datasets that are not related to ImageNet (and are reasonably sized, based on your computing budget). Pretrain models on this/these dataset(s) using the selected SSL methods. Implement so-called linear benchmark evaluation procedures on a subset of ImageNet (ImageNet100, imagenette or tiny-imagenet). Compare the pretrained models on this benchmark (including publicly available models trained on ImageNet), also analyze computational costs, training times, sensitivity to hyperparameters, etc.

Related materials:

- https://www.kaggle.com/datasets/ambityga/imagenet100
- https://huggingface.co/datasets/zh-plus/tiny-imagenet
- https://github.com/fastai/imagenette
- https://github.com/lightly-ai/lightly
- https://github.com/vturrisi/solo-learn
- https://arxiv.org/abs/2103.01988
- https://arxiv.org/abs/2103.13559
- https://proceedings.neurips.cc/paper/2020/hash/22f791da07b0d8a2504c2537c560001c-Abstract.html

Difficulty: hard

## Milestone 3

### How to run the pipeline

You should be running the repository locally to properly run the pipeline. Use `pip` or `miniconda` to install the dependencies.

#### Data download and preparation

Run all cells in the `notebooks/Data_preparation.ipynb` file to download and prepare the data into the `data` folder in the root.

### Train and test the models

Run the `notebooks/Training_and_testing.ipynb` file to train and test the models on the downloaded dataset. You can also load in a checkpoint from a commented out cell.

## Installation

Install required dependencies from `requirements.txt` file:

```bash
# with pip
pip install -r requirements.txt
```

The model training is limited to Jupiter Notebooks only, so container cannot be used. The files submitted for the Milestone 1 task are the `kaggle.json` and `notebooks/Data_preparation.ipynb` files located in the root folder. Open the `Data_preparation.ipynb` file and run the code. This should download the dataset to `data` folder, which the `notebooks/Training_and_testing.ipynb` Notebook can use to train and test the models with.

### Preparation

After downloading and extracting the files, open the `kaggle.json` file (located in the root directory). Replace the <> placeholders with your own username-API key pair, and then save the changes. To obtain a Kaggle API key, log in to your [Kaggle](https://www.kaggle.com/) account and in the settings menu, generate a new kaggle.json file containing the API token. Replace the values in the `kaggle.json` file.

The project contains a `Dockerfile` you can run to train and evaluate models. You can also use the commands below to install the project locally.

### Docker

When running with docker, don't forget to

- lower `num_workers` in `data/` configs to `0`.
- set `persistent_workers` in `data/` configs to `False`.

This way, DataLoader won't try to use shared memory and won't crash due to safety restrictions applied by docker.

You can also use `--ipc=host` flag when running docker to allow the container to use the host's shared memory. When using this flag, you don't need to modify `data/` configs.

```bash
# build docker image
docker build -t ssl-in-the-wild .

# run docker image (trains and evaluates models)
docker run ssl-in-the-wild --ipc=host
```

### Pip

```bash
# clone project
git clone https://github.com/HLCaptain/ssl-in-the-wild
cd ssl-in-the-wild

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

### Conda

```bash
# clone project
git clone https://github.com/HLCaptain/ssl-in-the-wild
cd ssl-in-the-wild

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train VICREG
python src/train.py

# train Classifier with VICREG backbone and Optuna Hyperparameter Optimzation
python src/train.py data=classifier model=classifier callbacks=classifier_train_callback hparams_search=classifier_optuna

# train Classifier with VICREG backbone and frozen weights
python src/train.py data=classifier model=classifier_ssl_frozen callbacks=classifier_ssl_frozen_train_callback

# train Classifier with VICREG backbone and NOT frozen weights
# python src/train.py data=classifier model=classifier_ssl_not_frozen callbacks=classifier_ssl_not_frozen_train_callback

# train Classifier without SSL, but pretrained backbone and frozen weights
# python src/train.py data=classifier model=classifier_pretrained_frozen callbacks=classifier_pretrained_frozen_train_callback

# train Classifier without SSL, but pretrained backbone and NOT frozen weights
# python src/train.py data=classifier model=classifier_pretrained_not_frozen callbacks=classifier_pretrained_not_frozen_train_callback

# train Classifier with untrained backbone
# python src/train.py data=classifier model=classifier_untrained callbacks=classifier_untrained_train_callback

# evaluate model with VICREG backbone and frozen weights on test dataset
python src/eval.py +callbacks=classifier_ssl_frozen_eval_callback
# evaluate other models

# python src/eval.py model=classifier_ssl_not_frozen +callbacks=classifier_ssl_not_frozen_eval_callback ckpt_path=models/best_model_with_ssl_not_frozen.ckpt
# python src/eval.py model=classifier_pretrained_frozen +callbacks=classifier_pretrained_frozen_eval_callback ckpt_path=models/best_model_with_pretrained_frozen.ckpt
# python src/eval.py model=classifier_pretrained_not_frozen +callbacks=classifier_pretrained_not_frozen_eval_callback ckpt_path=models/best_model_with_pretrained_not_frozen.ckpt
# python src/eval.py model=classifier_untrained +callbacks=classifier_untrained_eval_callback ckpt_path=models/best_model_with_untrained.ckpt
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## Contribution

Feel free to open an issue for any bugs or feature requests. See more in [CONTRIBUTING.md](CONTRIBUTING.md).

## Frameworks used

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

## Project Structure

The directory structure of new project looks like this:

```
├── .github                   <- Github Actions workflows
│
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Data scripts
│   ├── models                   <- Model scripts
│   ├── utils                    <- Utility scripts
│   │
│   ├── eval.py                  <- Run evaluation
│   └── train.py                 <- Run training
│
├── tests                  <- Tests of any kind
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── .project-root             <- File for inferring the position of project root directory
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`
├── pyproject.toml            <- Configuration options for testing and linting
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package
└── README.md
```

## Main Ideas

- [**Rapid Experimentation**](#your-superpowers): thanks to hydra command line superpowers
- [**Minimal Boilerplate**](#how-it-works): thanks to automating pipelines with config instantiation
- [**Main Configs**](#main-config): allow you to specify default training configuration
- [**Experiment Configs**](#experiment-config): allow you to override chosen hyperparameters and version control experiments
- [**Workflow**](#workflow): comes down to 4 simple steps
- [**Experiment Tracking**](#experiment-tracking): Tensorboard, W&B, Neptune, Comet, MLFlow and CSVLogger
- [**Logs**](#logs): all logs (checkpoints, configs, etc.) are stored in a dynamically generated folder structure
- [**Hyperparameter Search**](#hyperparameter-search): simple search is effortless with Hydra plugins like Optuna Sweeper
- [**Tests**](#tests): generic, easy-to-adapt smoke tests for speeding up the development
- [**Continuous Integration**](#continuous-integration): automatically test and lint your repo with Github Actions
- [**Best Practices**](#best-practices): a couple of recommended tools, practices and standards

## ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac
python train.py trainer=mps
```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer=gpu +trainer.precision=16
```

</details>

<!-- deepspeed support still in beta
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like W&B or Tensorboard</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).

> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.

> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) doesn't require you to add any boilerplate to your code, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml).

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> **Note**: This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

Update pre-commit hook versions in `.pre-commit-config.yaml` with:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Run tests</b></summary>

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Note**: You might need to escape the bracket characters in your shell with `python train.py tags=\["mnist","experiment_X"\]`.

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

## How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: src.models.mnist_model.MNISTLitModule
lr: 0.001
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 10
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file.

Switch between models and datamodules with command line arguments:

```bash
python train.py model=mnist
```

Example pipeline managing the instantiation logic: [src/train.py](src/train.py).

## Main Config

Location: [configs/train.yaml](configs/train.yaml)

Main project config contains default training configuration.

It determines how config is composed when simply executing command `python train.py`.

<details>
<summary><b>Show main project config</b></summary>

```yaml
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - data: mnist.yaml
  - model: mnist.yaml
  - callbacks: default.yaml
  - logger: null # set logger here or use command line (e.g. `python train.py logger=csv`)
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and datamodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

# task name, determines output directory path
task_name: "train"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
# appending lists from command line is currently not supported :(
# https://github.com/facebookresearch/hydra/issues/1547
tags: ["dev"]

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# simply provide checkpoint path to resume training
ckpt_path: null

# seed for random number generators in pytorch, numpy and python.random
seed: null
```

</details>

## Experiment Config

Location: [configs/experiment](configs/experiment)

Experiment configs allow you to overwrite parameters from main config.

For example, you can use them to version control best hyperparameters for each combination of model and dataset.

<details>
<summary><b>Show example experiment config</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /data: mnist.yaml
  - override /model: mnist.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["mnist", "simple_dense_net"]

seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "mnist"
```

</details>

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```

**Experiment design**

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python train.py -m logger=csv data.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```
2. Write a script or notebook that searches over the `logs/` folder and retrieves csv logs from runs containing given tags in config. Plot the results.

## Logs

Hydra creates new output directory for every executed run.

Default logging structure:

```
├── logs
│   ├── task_name
│   │   ├── runs                        # Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── csv                     # Csv logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   └── ...
│   │   │
│   │   └── multiruns                   # Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   ├──1                        # Multirun job number
│   │       │   ├──2
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                          # Logs generated when debugging config is attached
│       └── ...
```

</details>

You can change this structure by modifying paths in [hydra configuration](configs/hydra).

## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights&amp;Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_module.py).

## Tests

Template comes with generic tests implemented with `pytest`.

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

Most of the implemented tests don't check for any specific output - they exist to simply verify that executing some commands doesn't end up in throwing exceptions. You can execute them once in a while to speed up the development.

Currently, the tests cover cases like:

- running 1 train, val and test step
- running 1 epoch on 1% of data, saving ckpt and resuming for the second epoch
- running 2 epochs on 1% of data, with DDP simulated on CPU

And many others. You should be able to modify them easily for your use case.

There is also `@RunIf` decorator implemented, that allows you to run tests only if certain conditions are met, e.g. GPU is available or system is not windows. See the [examples](tests/test_train.py).

## Hyperparameter Search

You can define hyperparameter search by adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/acc_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      data.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
      model.net.lin2_size: choice(64, 128, 256)
      model.net.lin3_size: choice(32, 64, 128, 256)
```

</details>

Next, execute it with: `python train.py -m hparams_search=mnist_optuna`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/task_name/multirun` folder.

This approach doesn't support resuming interrupted search and advanced techniques like prunning - for more sophisticated search and workflows, you should probably write a dedicated optimization task (without multirun feature).

## Continuous Integration

Template comes with CI workflows implemented in Github Actions:

- `.github/workflows/test.yaml`: running all tests with pytest
- `.github/workflows/code-quality-main.yaml`: running pre-commits on main branch for all files
- `.github/workflows/code-quality-pr.yaml`: running pre-commits on pull requests for modified files only

## Distributed Training

Lightning supports multiple ways of doing distributed training. The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read the [lightning docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

You can run DDP on mnist example with 4 GPUs like this:

```bash
python train.py trainer=ddp
```

> **Note**: When using DDP you have to be careful how you write your models - read the [docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

## Accessing Datamodule Attributes In Model

The simplest way is to pass datamodule attribute directly to model on initialization:

```python
# ./src/train.py
datamodule = hydra.utils.instantiate(config.data)
model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```

> **Note**: Not a very robust solution, since it assumes all your datamodules have `some_param` attribute available.

Similarly, you can pass a whole datamodule config as an init parameter:

```python
# ./src/train.py
model = hydra.utils.instantiate(config.model, dm_conf=config.data, _recursive_=False)
```

You can also pass a datamodule config parameter to your model through variable interpolation:

```yaml
# ./configs/model/my_model.yaml
_target_: src.models.my_module.MyLitModule
lr: 0.01
some_param: ${data.some_param}
```

Another approach is to access datamodule in LightningModule directly through Trainer:

```python
# ./src/models/mnist_module.py
def on_train_start(self):
  self.some_param = self.trainer.datamodule.some_param
```

> **Note**: This only works after the training starts since otherwise trainer won't be yet available in LightningModule.

## Best Practices

<details>
<summary><b>Use Miniconda</b></summary>

It's usually unnecessary to install full anaconda environment, miniconda should be enough (weights around 80MB).

Big advantage of conda is that it allows for installing packages without requiring certain compilers or libraries to be available in the system (since it installs precompiled binaries), so it often makes it easier to install some dependencies e.g. cudatoolkit for GPU support.

It also allows you to access your environments globally which might be more convenient than creating new local environment for every project.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Update conda:

```bash
conda update -n base -c defaults conda
```

Create new conda environment:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.

Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.

To reformat all files in the project use command:

```bash
pre-commit run -a
```

To update hook versions in [.pre-commit-config.yaml](.pre-commit-config.yaml) use:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy

class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```
2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def on_train_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def on_validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def on_test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.

To initialize the dvc repository:

```bash
dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
dvc add data/MNIST
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:

```bash
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and complete the `setup.py` file.

Now your project can be installed from local files:

```bash
pip install -e .
```

Or directly from git repository:

```bash
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```

So any file can be easily imported into any other file like so:

```python
from project_name.models.mnist_module import MNISTLitModule
from project_name.data.mnist_datamodule import MNISTDataModule
```

</details>

<details>
<summary><b>Keep local configs out of code versioning</b></summary>

Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file [configs/local/default.yaml](configs/local/) can be created which is automatically loaded but not tracked by Git.

For example, you can use it for a SLURM cluster config:

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

## Resources

This template was inspired by:

- [PyTorchLightning/deep-learning-project-template](https://github.com/PyTorchLightning/deep-learning-project-template)
- [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
- [lucmos/nn-template](https://github.com/lucmos/nn-template)

Other useful repositories:

- [jxpress/lightning-hydra-template-vertex-ai](https://github.com/jxpress/lightning-hydra-template-vertex-ai) - lightning-hydra-template integration with Vertex AI hyperparameter tuning and custom training job
