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
```
== Configuration groups ==
Compose your configuration from those groups (group=option)

dm: celeba, cmnist
dm/celeba: male_blond, male_smiling
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

### Different dataset
To explore the different datasets, you can try
```bash
python main.py dm=celeba --help
```
You can also set
```bash
python main.py dm=celeba/male_smiling
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

## Structure of the code

- `main.py`: The main entry point of the code. It sets up Hydra and then calls the main `run()` function.
- `src/`
  - `run.py`: Contains the main `Config` class that is used to define valid config values. It also contains the `run()` function that is called by `main.py`.
  - `data.py`: Contains the `DataModule` class that is used to load the data.
  - `model.py`: Contains the `ModelFactory` class that is used to create the model.
  - `logging.py`: Contains the `WandbCfg` class that is used to set up Weights & Biases logging.
- `conf/`
  - `config.yaml`: The main config file for the project. It sets the default values for `dm` and `model`.
  - `hydra/`
    - `launcher/`
      - `slurm/`: Contains the SLURM launcher config files.
  - `dm/`: Contains the config files for the different datasets.
  - `model/`: Contains the config files for the different model architectures.
