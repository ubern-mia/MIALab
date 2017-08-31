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

1. Launch the installer by double clicking

2. Click Next

3. Agree the license

4. Select an install "Just Me" and click Next

5. Select an install directory and click Next

6. Choose whether to add Anaconda to your PATH environment variable. We recommend not adding Anaconda to the PATH environment variable, since this can interfere with other software.

7. Choose whether to register Anaconda as your default Python 3.6. Unless you plan on installing and running multiple versions of Anaconda, or multiple versions of Python, you should accept the default and leave this box checked.

8. Click Install

9. Click Next

10. Unselect the checkboxes and lick Finish

11. Verify the installation by opening the Anaconda Navigator

Linux
------
Run the following commands in the terminal (tested on ubuntu 16.04 LTS).

1. Create an Anaconda install directory

    - :bash:`sudo mkdir /usr/local/anaconda`

2. Install Anaconda

    - :bash:`~/Downloads/Anaconda3-4.4.0-Linux-x86_64.sh` (replace :bash:`~/Downloads/` with the path to the installer file)
    - Scroll to the bottom of the license and enter :bash:`yes` to agree the license
    - :bash:`/usr/local/anaconda/anaconda3` to change the install directory
    - :bash:`yes` to add Anaconda to the PATH

3. Change the install directory permission

    - :bash:`sudo chown -R usr /usr/local/anaconda`

4. Verify the installation by :bash:`conda list`, which should list all installed Anaconda packages

5. Create a new Python 3.6 environment with the name mialab

    - :bash:`conda create -n mialab python=3.6`

6. Activate the environment by :bash:`source activate mialab`

7. Install all required packages for the MIALab

    - :bash:`cd /path/to/MIALab/repository`
    - :bash:`python setup.py install` will install all required packages

8. Execute the hello world to verify the installation

    - :bash:`python ./bin/hello_world.py`

9. Run Sphinx to create the documentation

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
