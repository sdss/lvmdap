Installation
============

Prerequisites
-------------

LVM-DAP depends on **pyPipe3D** (`pyPipe3D <http://ifs.astroscu.unam.mx/pyPipe3D/>`_, Lacerda et al. 2022).
Install pyPipe3D before proceeding with LVM-DAP installation.

Recommended Setup
-----------------

It is recommended to install LVM-DAP in a virtual environment to avoid dependency issues. Options include:

* `Conda/Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (Recommended)
* `venv <https://docs.python.org/3.8/library/venv.html>`_
* `pipenv <https://pipenv.pypa.io/en/latest/>`_

Installation Steps
------------------

Using Conda (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create and activate a Conda environment
    conda create --name lvmdap python=3.11
    conda activate lvmdap

    # Install specific matplotlib version (required)
    pip install matplotlib==3.7.3

    # Clone and install LVM-DAP
    git clone https://github.com/sdss/lvmdap.git
    cd lvmdap
    pip install . --user

Using Poetry
~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/sdss/lvmdap.git
    cd lvmdap

    # Install with poetry
    poetry install

Additional Data
---------------

Download and store the required fitting data (stellar templates) from:
`Download LVM fitting data <https://tinyurl.com/mudr6yw7>`_

Place the downloaded data in the ``_fitting_data/`` directory.

Environment Variables
---------------------

Define the following environment variables (recommended):

.. code-block:: bash

    export LVM_DAP="/path/to/lvmdap"           # LVM-DAP installation directory
    export LVM_DAP_CFG="$LVM_DAP/_legacy"      # Configuration files directory
    export LVM_DAP_RSP="_fitting_data"         # Stellar templates directory

Verifying Installation
----------------------

Test your installation by running:

.. code-block:: bash

    lvm-dap-conf _examples/data/lvmSFrame-example.fits.gz test-run _legacy/lvm-dap_fast.yaml

Troubleshooting
---------------

* If you encounter installation issues, check **numpy version** compatibility
* Ensure **pyPipe3D** is installed **before** LVM-DAP
* The specific matplotlib version (3.7.3) is required for compatibility
* For support, refer to the `official repository <https://github.com/sdss/lvmdap>`_
