# Gravlax: Basic training utils for [JAX]

[**Overview**](#overview)
| [**Installation?**](#installation)

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

[JAX]: https://github.com/google/jax
