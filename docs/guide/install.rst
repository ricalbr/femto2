Installation guide
==================

``femto`` can be installed using ``pip``.

As a first step, check that you have a working Python with ``pip`` installed by running the following command:

.. code-block:: bash

    python -m pip --version
..

If ``pip`` is not installed in the system, follow this `installation guide <https://pip.pypa.io/en/stable/installation/>`_.

****

The preferred way to install the package is using a virtual environment. In Python this can be done using several tool such as virtualenv_ or conda_.


.. _virtualenv:

virtualenv
~~~~~~~~~~
``virtualenv`` has to be installed from ``pip``


.. _conda:

conda/miniconda/Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~
Open the conda/Anaconda prompt and type the following commands

- Update the conda manager:

.. code-block:: bash

    conda update conda
..

- Create a virtual environment:

.. code-block:: bash

    conda create --name femto python=3.8
..

.. note::

    ``femto`` supports all the versions of Python from 3.7.

- Activate the environment:

.. code-block:: bash

    conda activate femto
..

****

Once the virtual environment is set up

- Clone the remote repository locally to your computer:

.. code-block:: bash

    git clone git@github.com:ricalbr/femto.git
..

    Alternatively, the repository can also be cloned directly from Gihub website.


- Change the directory:

.. code-block:: bash

    cd femto
..

- Install the packages (and its dependencies) with this command:

.. code-block:: bash

    pip install -e .
..
