.. _installation_label:

Installation
=============

.. role:: bash(code)
   :language: bash

To start with the installation, download the `Anaconda installer <https://www.anaconda.com/distribution/>`_ for your operating system and Python 3.7.


Windows
--------
The installation has been tested on Windows 10.

#. git installation
   
   - Download `git <https://git-scm.com/downloads>`_ and install

#. Clone MIALab repository
   
   - open "Git Bash"
   - :bash:`cd \path\to\where\you\want\the\code`
   - :bash:`git clone https://github.com/ubern-mia/MIALab.git`

#. Anaconda installation

   - Follow the instructions on the `official website <https://docs.anaconda.com/anaconda/install/windows/>`__

#. Verify the installation
   
   - Open "Anaconda Prompt"
   - :bash:`conda list`, which should list all installed Anaconda packages

#. Create a new Python 3.7 environment with the name mialab
   
   - :bash:`conda create -n mialab python=3.7`

#. Activate the environment by
   
   - :bash:`activate mialab`

#. Install all required packages for the MIALab
   
   - :bash:`cd /d \path\to\MIALab\repository`
   - :bash:`pip install -r requirements.txt` will install all required packages

#. Execute the hello world to verify the installation
   
   - :bash:`python .\bin\hello_world.py`

#. Run Sphinx to create the documentation
   
   - :bash:`sphinx-build -b html .\docs .\docs\_build`
   - The documentation is now available under ``.\docs\_build\index.html``

Linux
------
Run the following commands in the terminal (tested on Ubuntu 16.04 LTS and 18.04 LTS).

#. git installation
   
   - :bash:`sudo apt-get install git`

#. Clone MIALab repository
   
   - :bash:`cd /path/to/where/you/want/the/code`
   - :bash:`git clone https://github.com/ubern-mia/MIALab.git`

#. Run Anaconda installation script

   - Follow the instructions on the `official website <https://docs.anaconda.com/anaconda/install/linux>`__
   - No need to install the GUI packages

#. Verify the installation
   
   - :bash:`conda list`, which should list all installed Anaconda packages

#. Create a new Python 3.7 environment with the name mialab (confirm with y when promted during creation)
   
   - :bash:`conda create -n mialab python=3.7`

#. Activate the environment by
   
   - :bash:`conda activate mialab`

#. Install all required packages for the MIALab
   
   - :bash:`cd /path/to/MIALab/repository`
   - :bash:`pip install -r requirements.txt` will install all required packages

#. Execute the hello world to verify the installation
   
   - :bash:`python ./bin/hello_world.py`

#. Run Sphinx to create the documentation
   
   - :bash:`sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under ``./docs/_build/index.html``


macOS
------
The installation has not been tested.

#. git installation
   
   - Download `git <https://git-scm.com/downloads>`_ and install

#. Clone MIALab repository
   
   - :bash:`cd /path/to/where/you/want/the/code`
   - :bash:`git clone https://github.com/ubern-mia/MIALab.git`

#. Anaconda installation

   - Follow the instructions on the `official website <https://docs.anaconda.com/anaconda/install/mac-os/>`__

#. Verify the installation
   
   - :bash:`conda list`, which should list all installed Anaconda packages

#. Create a new Python 3.7 environment with the name mialab
   
   - :bash:`conda create -n mialab python=3.7`

#. Activate the environment by
   
   - :bash:`source activate mialab`

#. Install all required packages for the MIALab
   
   - :bash:`cd /path/to/MIALab/repository`
   - :bash:`pip install -r requirements.txt` will install all required packages

#. Execute the hello world to verify the installation
   
   - :bash:`python ./bin/hello_world.py`
 
#. Run Sphinx to create the documentation
   
   - :bash:`sphinx-build -b html ./docs ./docs/_build`
   - The documentation is now available under ``./docs/_build/index.html``
