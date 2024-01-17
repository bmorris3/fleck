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

.. note::

    Known issue for M2 Macs: as of January 2024, pip will install a version of jaxlib 
    that may not work, raising the following error::

        RuntimeError: This version of jaxlib was built using AVX instructions, which your CPU and/or operating system do not support.

    The easiest workaround is to uninstall this version of jax with::

        pip uninstall jax jaxlib
    
    and then install jax via conda::

        conda install -c conda-forge jaxlib
        conda install -c conda-forge jax
    
