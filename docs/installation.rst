=============
Installation
=============

.. role:: bash(code)
   :language: bash

To start with the installation,

1. Download the `Anaconda installer <https://www.anaconda.com/download/>`_ for your operating system and Python 3.6.

2. Clone this MIALab repository

You can follow our install instructions here but for Anaconda instructions are available `online <https://docs.continuum.io/anaconda/install/>`_ as well.

Windows
--------
The installation has been tested on Windows 10.

#. Anaconda installation (`official website <https://docs.anaconda.com/anaconda/install/windows.html>`_)
   
   - Launch the installer
   - Select an install for “Just Me” unless you’re installing for all users (which requires Windows Administrator privileges)
   - Choose whether to add Anaconda to your PATH environment variable. We recommend not adding Anaconda to the PATH environment variable, since this can interfere with other software.
   - Choose whether to register Anaconda as your default Python 3.6. Unless you plan on installing and running multiple versions of Anaconda, or multiple versions of Python, you should accept the default and leave this box checked.

#. Verify the installation by opening the Anaconda Navigator
   
#. TODO create environment
   


Linux
------
Run the following commands in the terminal (tested on ubuntu 16.04 LTS).

#. Run Anaconda installation script (`official website <https://docs.anaconda.com/anaconda/install/linux>`_)
   
   - :bash:`bash <path_to_file>/Anaconda3-4.4.0-Linux-x86_64.sh` (run the installation script)
     
     - Scroll to the bottom of the license and enter :bash:`yes` to agree the license
     - Accept suggested installation path (or change it if you know what you do)
     - :bash:`yes` to add Anaconda to the PATH
     - Reopen the terminal

#. Verify the installation
   
   - :bash:`conda list`, which should list all installed Anaconda packages

#. Create a new Python 3.6 environment with the name mialab
   
   - :bash:`conda create -n mialab python=3.6`

#. Activate the environment by
   
   - :bash:`source activate mialab`

#. Install all required packages for the MIALab
   
   - :bash:`cd /path/to/MIALab/repository`
   - :bash:`pip install .` will install all required packages

#. Execute the hello world to verify the installation
   
   - :bash:`python ./bin/hello_world.py`

#. Run Sphinx to create the documentation
   
   - :bash:`cd sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under :bash:`/docs/_build/index.html`


macOS
------
The installation has not been tested.

#. Anaconda installation (`official website <https://docs.anaconda.com/anaconda/install/mac-os>`_)
   
   - Launch the installer
   - On the Destination Select screen, select "Install for me only"

#. Verify the installation by opening the Anaconda Navigator

#. TODO create environment
