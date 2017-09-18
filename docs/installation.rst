=============
Installation
=============

.. role:: bash(code)
   :language: bash

To start with the installation,

1. Download the `Anaconda installer <https://www.anaconda.com/download/>`_ for your operating system and Python 3.6.

2. Download the MIALab repository content and extract it

You can follow our install instructions here but for Anaconda instructions are available `online <https://docs.continuum.io/anaconda/install/>`_ as well.

Windows
--------
The installation has been tested on Windows 10.

#. Launch the installer by double clicking

#. Click Next

#. Agree the license

#. Select an install "Just Me" and click Next

#. Select an install directory and click Next

#. Choose whether to add Anaconda to your PATH environment variable. We recommend not adding Anaconda to the PATH environment variable, since this can interfere with other software.

#. Choose whether to register Anaconda as your default Python 3.6. Unless you plan on installing and running multiple versions of Anaconda, or multiple versions of Python, you should accept the default and leave this box checked.

#. Click Install

#. Click Next

#. Unselect the checkboxes and lick Finish

#. Verify the installation by opening the Anaconda Navigator

Linux
------
Run the following commands in the terminal (tested on ubuntu 16.04 LTS).

#. Make script executable 
   
   - :bash:`chmod +x <path_to_file>/Anaconda3-4.4.0-Linux-x86_64.sh` (make file executable)

#. Run Anaconda installation script
   
   - :bash:`<path_to_file>/Anaconda3-4.4.0-Linux-x86_64.sh` (run the installation script)
     
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
The installation has been tested on todo(fabianbalsiger).

1. Launch the installer by double clicking

2. Answer the prompts on the Introduction, Read Me, and License screens

3. Select Install for me only

4. Change the install directory if you want (by default Anaconda installs in your home directory)

5. Click Install

6. Click Close
