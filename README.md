# :sushi: Gravlax: Basic training utils for [JAX]

| [**Overview**](#overview)
| [**Installation?**](#installation)
| [**Examples**](#examples)
|

> [!IMPORTANT]
> This library is very much a work in progress. Much of the code contained within is used across various research projects and is subject to change.
> Please see sections below for basic usecases.

## Overview<a id="overview"></a>

[JAX] is a numerical computing library that combines NumPy, automatic differentiation, and first-class GPU/TPU support.

Gravlax is a set of basic utilies which may be useful if you use [JAX] in your day-to-day research. At the moment, the code largely resembles basic snippets used across a multitude of projects. Building this library provides a form of centralisation for these utilities.

## Installation<a id="installation"></a>

Gravlax is easily installable via pip:

```bash
$ pip install gravlax
```

If you wish to install for development purposes, clone the repo and run:

```bash
$ pip install -e ".[dev]"
```

and be sure to install the `pre-commit`, as per:

```bash
$ pre-commit install
```

This should allow you to get up and running fairly quickly.

## Examples<a id="examples"></a>

When training a model, we often employ a mini-batching approach, recording statistics for each batch. When logging performance metrics, we often want to report a quantity representative of the batch, often the mean across said batch. It is also nice to have an idea of how long each batch takes to run.

In order to address these problems, `gravlax` provides the `BatchManager` context manager:

```python
import gravlax as glx

# ... code here

with glx.BatchManager(name='training', n_batches=32) as bm_train:

  for batch in trainloader:
    loss_dict, state = train_step(state, batch)
    bm_train.register_loss(loss_dict)

with glx.BatchManager(name='training', n_batches=32) as bm_validation:

  for batch in validationloader:
    loss_dict, _ = validation_step(state, batch)
    bm_validation.register_loss(loss_dict)
```

Now metrics have been recorded across the batch, we can reduce these for the purpose of logging to file or `stdout`. Gravlax provides a nice `write_dict_to_csv` function which ensures headers are written correctly to file, based on the keys in the dictionary. Dictionary keys are augmented with the name assigned to the `BatchManager`, prepending the name to the keys.

We can reduce the recorded losses and write to csv as follows:


```python
csv_dict = bm_train.reduce() | bm_validation.reduce()
glx.write_dict_to_csv(CSV_PATH, csv_dict)
```

We can also use the `BatchManager` instances to record the time taken to process all batches:

```python
print(f'[Timing] Train = {bm_train.time}s ; Validation = {bm_validation.time}s')
```

Additional functionality will be coming soon -- these really just show the basics available at the moment.

[JAX]: https://github.com/google/jax
