# An annotated example of using Hydra

This code base in meant to show-case how to use hydra to manage configuration in a project.

## Installing dependencies

If you have `rye` installed:
```bash
rye sync --no-lock
```

If not, run (after activating an appropriate python env):
```bash
pip install -r requirements.lock
```

## Running the code

To run the code with default configs, simply do:
```bash
python main.py
```

The output should be something like this:
```
Starting a run with seed 42 and GPU 0.
Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
```

We can set the seed and the GPU manually with the `seed` and `gpu` arguments:
```bash
python main.py seed=123 gpu=1
```

To find out which arguments are available, run:
```bash
python main.py --help
```

It should look like this:

<details>
<summary> Help output (quite long) </summary>

```
== Configuration groups ==
Compose your configuration from those groups (group=option)

dm: celeba, celeba_male_blond, celeba_male_smiling, celeba_male_smiling_small, cmnist
experiment: good_run
experiment/cmnist: cnn
model: cnn, fcn, single_linear_layer


== Config ==
Override anything in the config (foo.bar=value)

dm:
  root: !!python/object/apply:pathlib.PosixPath
  - /
  - srv
  - galene0
  - shared
  - data
  default_res: 28
  label_map: null
  colors: null
  num_colors: 10
  val_prop: 0.2
model:
  num_hidden: 1
  hidden_dim: null
  activation: GELU
  norm: LN
  dropout_prob: 0.0
  final_bias: true
  input_norm: false
opt:
  lr: 5.0e-05
  weight_decay: 0.0
  optimizer_cls: torch.optim.AdamW
  optimizer_kwargs: null
wandb:
  name: null
  dir: ./local_logging
  id: null
  anonymous: null
  project: hydra-example
  group: null
  entity: predictive-analytics-lab
  tags: null
  job_type: null
  mode: disabled
  resume: null
seed: 42
gpu: 0
```

</details>

We can see that `seed` and `gpu` are "top-level" config values, but there are also a lot of values in subconfigs. For example, we can change the `num_hidden` value in the `model` config:
```bash
python main.py model.num_hidden=2
```
This should print a different model architecture.

But we can also see that there are different "schemas" available in the "model" group:
```
model: cnn, fcn, single_linear_layer
```
This allows us to completely change the model architecture:
```bash
python main.py model=cnn
```

With the "cnn" architecture, the config key `model.hidden_dim` is not available anymore, so the following should raise an error:
```bash
python main.py model=cnn model.hidden_dim=10
```
Output:
```
Could not override 'model.hidden_dim'.
To append to your config use +model.hidden_dim=10
Key 'hidden_dim' not in 'SimpleCNNFactory'
    full_key: model.hidden_dim
    reference_type=ModelFactory
    object_type=SimpleCNNFactory
```

To see the available options for the "cnn" architecture, you can run:
```bash
python main.py model=cnn --help
```
Now the "model" part of the output should look like this:
```
model:
  kernel_size: 5
  pool_stride: 2
  activation: RELU
```
This tells us that we can set the `kernel_size`, `pool_stride`, and `activation` values for the "cnn" architecture. For example:
```bash
python main.py model=cnn model.kernel_size=3 model.pool_stride=1 model.activation=SELU
```

### W&B config
To enable W&B logging, you can set the `wandb.mode` to "online" or "offline" (default is "disabled"):
```bash
python main.py wandb.mode=online
```

## Basics of config files
Let's say you are often working with a configuration of the CelebA dataset which looks like this:
```bash
python main.py dm=celeba dm.superclass=SMILING dm.subclass=MALE
```

(Once again, you can set `dm=celeba` and use `--help` to see the available options for the CelebA dataset.)

But after a while, it gets tiring to always type this out. Instead, you can create a config file in the `conf/dm/` directory to store often-used datamodule configurations. For example, you can create a file `conf/dm/celeba_male_smiling.yaml` with the following content:
```yaml
defaults:
  - celeba

superclass: SMILING
subclass: MALE
```

You can think of the `defaults` entry as specifying inheritance.

Subsequently, you can use the config file like this:
```bash
python main.py dm=celeba_male_smiling
```

You can still override any values on the command line, just as before:
```bash
python main.py dm=celeba_male_smiling dm.superclass=BLOND dm.default_res=64
```

And you can create config files which inherit from other config files you created. For example, you can create a file `conf/dm/celeba_male_smiling_small.yaml` with the following content:
```yaml
defaults:
  - celeba_male_smiling

default_res: 64
```

## Multiruns
Often, we want to run the same experiment with only slightly different configurations. This can be done with the `--multirun` flag. For example, to run the code with seeds 42 and 43, we can do:
```bash
python main.py --multirun seed=42,43
```
The output should look like this:
```
[2024-05-16 18:40:47,580][HYDRA] Launching 2 jobs locally
[2024-05-16 18:40:47,581][HYDRA]        #0 : seed=42
Starting a run with seed 42 and GPU 0.
Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
[2024-05-16 18:40:47,672][HYDRA]        #1 : seed=43
Starting a run with seed 43 and GPU 0.
Model architecture:
Sequential(
  (0): Sequential(
    (0): Linear(in_features=28, out_features=28, bias=True)
    (1): LayerNorm((28,), eps=1e-05, elementwise_affine=True)
    (2): GELU(approximate='none')
  )
  (1): Linear(in_features=28, out_features=10, bias=True)
)
```

By default, Hydra simply runs the code with the different config values sequentially. But if you have access to a SLURM cluster, you can also run the jobs in parallel. To do this, we need to set `hydra/launcher=...` to the appropriate value. In this code base, there is a config file at `conf/hydra/launcher/slurm/kyiv.yaml` that has the correct settings for the Kyiv machine. To use it, we can run:
```bash
python main.py --multirun seed=42,43 hydra/launcher=slurm/kyiv
```

Hydra doesn't limit you to iterating over just one parameter. You can also iterate over multiple parameters. For example, to run the code with seeds 42 and 43 and `model.num_hidden` set to 1, 2 and 3, you can do:
```bash
python main.py --multirun seed=42,43 model.hidden_dim=10 model.num_hidden=1,2,3 gpu=0
```
This will start 6 jobs: 2 seeds x 3 `model.num_hidden` values.

## Hyperparameter search
If you don't want to just blindly try out model parameters, you can also use Hydra to perform hyperparameter search with libraries like Optuna, Nevergrad, and Ax.

In order to use any of these libraries, you need to return a value from the function decorated with `@hydra.main(...)` (located in `main.py`). The value should be a `float` and should represent the metric you want to optimize. For example, if you want to optimize the accuracy of the model, you can return the validation accuracy. If you want to minimize the loss, you can return the validation loss.

To use Optuna, for example, create a config file in `conf/hydra/sweeper/`. The name doesn't matter, but we'll choose `fcn_params.yaml` and fill the content like this:
```yaml
defaults:
  # Select here which hyperparameter optimization library to use.
  - optuna

sampler:
  warn_independent_sampling: true

study_name: fcn_params
n_trials: 25
n_jobs: 3
direction:
  # The direction depends on what you are returning from your main() function.
  - maximize

params:
  opt.lr: tag(log, interval(1.e-5, 1.e-3))
  model.num_hidden: range(1, 4)
  model.dropout_prob: choice(0.0, 0.1)
```

Let's break down what's going on here:

- First, we select the right `defaults` value. In this case, we want to use Optuna, so we set `optuna`.
- Then, some settings for Optuna follow. To see all available settings, check out the docs: https://hydra.cc/docs/plugins/optuna_sweeper/
- Finally, we specify `params`, which is the heart of this config file. Here, we specify the parameters we want to optimize. For each parameter, we have to specify the search space. For example, `opt.lr: tag(log, interval(1.e-5, 1.e-3))` means that we want to optimize the `opt.lr` parameter and we want to search in the log space between `1.e-5` and `1.e-3`. The `model.num_hidden: range(1, 4)` we want to search in the range from 1 to 3 (inclusive). The `model.dropout_prob: choice(0.0, 0.1)` means that we want to choose between 0.0 and 0.1.

## Global experiment configs
We already saw that you can create config files to, for example, save a common data module configuration – such a config file is then put in `conf/dm/`. However, such config files can only configure one subcomponent of the whole configuration. This means you might end up with long commands like this one:
```bash
python main.py model=single_linear_layer model.activation=SELU dm=celeba_male_smiling_small dm.download=false gpu=0 seed=1 opt.lr=0.001 opt.weight_decay=0.001
```
which you have to type over and over again.

Luckily, we can define "experiment configs" in the `conf/experiment/` directory, that act as *global* configurations. For the above command, we can create a file at `conf/experiment/good_run.yaml` with the following content:
```yaml
# @package _global_
---
defaults:
  - override /model: single_linear_layer
  - override /dm: celeba_male_smiling_small

seed: 1
gpu: 0

model:
  activation: SELU

dm:
  download: false

opt:
  lr: 0.001
  weight_decay: 0.001
```

Note that the comment `# @package _global_` is required. (The reason is that, by default, if you have a config file in the `conf/experiment/` directory, Hydra will want to associate this with the `experiment` entry in the main configuration – which doesn't exist! So, `@package _global_` tells Hydra to put the content of the file at the *top level* of the main config.)

And then we can run the code with these config values by running:
```bash
python main.py +experiment=good_run
```

You can still override values:
```bash
python main.py +experiment=good_run model.hidden_dim=20
```

**Note**: It's very easy to fall into the trap of defining *everything* in these global experiment configs, but that leads to lots of duplication. Try to put as much configuration as possible into the component-specific config files (i.e., those in `conf/dm/` and `/conf/model` and so on), because those configs are very easy to reuse across different experiments.

As the experiment configs are global, the directory structure doesn't matter at all. You can put a file into `conf/experiment/cmnist/cnn.yaml` and call it with
```bash
python main.py +experiment=cmnist/cnn
```
and it will work fine.

## How to make your code easily configurable with Hydra
What we found is the best method to structure your code to make it easy to configure with Hydra is the "builder" or "factory" pattern.

This means you have a dataclass that contains all the configuration values for a particular component of your code (e.g. the model, the data, the optimiser, etc.). And then this class has a `build()` or `init()` method that takes additional arguments which are only available at runtime (e.g. the input size of the model, the number of classes in the dataset, etc.), and then instantiates the component.

For example, in this code base, we have the `ModelFactory` class in `src/model.py`:
```python
@dataclass(eq=False)  # note that this should be a dataclass even though it has no fields
class ModelFactory(ABC):
    """Interface for model factories."""

    @abstractmethod
    def build(self, in_dim: int, *, out_dim: int) -> nn.Module:
        raise NotImplementedError()
```
And when you add a model to the code base, you subclass `ModelFactory` and implement the `build()` method:
```python
@dataclass(eq=False, kw_only=True)
class SimpleCNNFactory(ModelFactory):
    """Factory for a very simple CNN."""

    kernel_size: int = 5
    pool_stride: int = 2
    activation: Activation = Activation.RELU

    @override
    def build(self, in_dim: int, *, out_dim: int) -> nn.Sequential:
        # (implementation of the model)
```


## Structure of the code

- `main.py`: The main entry point of the code. It sets up Hydra and then calls the main `run()` function.
- `src/`
  - `run.py`: Contains the main `Config` class that is used to define valid config values. It also contains the `run()` function that is called by `main.py`.
  - `data.py`: Contains the `DataModule` class that is used to load the data.
  - `model.py`: Contains the `ModelFactory` class that is used to create the model.
  - `optimisation.py`: Contains the `OptimisationCfg` class that is used to build the optimiser that trains the model.
  - `logging.py`: Contains the `WandbCfg` class that is used to set up Weights & Biases logging.
- `conf/`
  - `config.yaml`: The main config file for the project. It sets the default values for `dm` and `model`.
  - `hydra/`
    - `launcher/`
      - `slurm/`: Contains the SLURM launcher config files.
  - `dm/`: Contains the config files for the different datasets.
  - `model/`: Contains the config files for the different model architectures.
