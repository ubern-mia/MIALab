Integrated Development Environment (IDE)
========================================

We recommend to use `JetBrains PyCharm <https://www.jetbrains.com/pycharm/>`_ as IDE to program in Python.
The community edition is open-source and sufficient for our purposes.
Follow the `instructions <https://www.jetbrains.com/help/pycharm/requirements-installation-and-launching.html>`_ to install PyCharm.

To open the MIALab as project and to configure the Python interpreter do the following:

#. Launch PyCharm
#. Click Open (or File > Open)

    #. In the dialog navigate to ``</path/to/where/you/have/the/code>/MIALab``
    #. Click OK
    #. MIALab is now open as PyCharm project (PyCharm created the ``.idea`` directory)

#. Click File > Settings... to open the settings dialog

    #. Navigate to Project: MIALab > Project Interpreter
    #. Select the Python interpreter ``</path/to/your/anaconda/installation>/envs/mialab/bin/python`` (on Linux and macOS) or ``<\path\to\your\anaconda\installation>\envs\mialab\python.exe`` (on Windows)

        - If the interpreter is not available in the combo box, click the gear icon and choose Add Local and navigate the the files above

    #. Confirm by clicking OK

#. Open the ``hello_world.py`` (``bin`` directory) in the navigator

    #. Right click in the editor > Run 'hello_world'
    #. Runs the hello_world and adds a configuration (see top right corner) to the project
    #. You can add configurations manually under Run > Edit Configurations...

You can watch the `getting started <https://www.jetbrains.com/pycharm/documentation/>`_ videos to get accustomed with the interface.

Additional Configuration
-------------------------

To change the **docstring** format to Google do the following:

#. Click File > Settings... to open the settings dialog
#. Navigate to Tools > Python Integrated Tools
#. Select Google in the Docstring format dropbox
#. Click OK

To add a configuration for the **Sphinx documentation** do the following:

#. Click Run > Edit Configurations...
#. Click Add New Configuration (plus icon) > Python docs > Sphinx task
#. Edit the following

    #. Name (e.g. ``docs``)
    #. Input to ``</path/to/where/you/have/the/code>/MIALab/docs`` (on Linux and macOS) or ``<\path\to\where\you\have\the\code>\MIALab\docs`` (on Windows)
    #. Output to ``</path/to/where/you/have/the/code>/MIALab/docs/_build/html`` (on Linux and macOS) or ``<\path\to\where\you\have\the\code>\MIALab\docs\_build\html`` (on Windows)

#. Click OK
