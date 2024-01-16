.. _install:

************
Installation
************

Install via pip
---------------

To install the most recent release of fleck (without JAX dependencies,
as published in `JOSS <https://doi.org/10.21105/joss.02103>`_), you may::

    python -m pip install fleck

Install from source
-------------------

Clone the repository, change directories into it, and build from source::

    git clone https://github.com/bmorris3/fleck.git
    cd fleck
    python -m pip install -e .

If you are ready to use `JAX <https://github.com/google/jax>`_ in the
``fleck.jax`` module, some additional dependencies are required, and you
can get those dependencies at install time with::

    python -m pip install -e .[jax]
