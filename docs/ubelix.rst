.. _ubelix_label:

.. role:: bash(code)
   :language: bash


UBELIX HPC
==========
The UBELIX (University of Bern Linux Cluster) is a HPC cluster of the University of Bern. During the MIALab course
you can use UBELIX for computing your experiments. Beside this short guide, we recommend reading the `official
documentation of UBELIX <https://hpc-unibe-ch.github.io/>`_.


.. important::
    The access to the UBELIX HPC is only granted to students officially enrolled at the University of Bern.


Activation & Installation
-------------------------
The general activation and installation procedure is independent on your operating system.
If you need assistance, please consult your lecturer.

#. Request `UBELIX access <https://hpc-unibe-ch.github.io/getting-Started/account.html>`_ with your student account

#. Install a SSH client on your machine (see :ref:`ssh_clients_label`)

#. Install a SFTP client on your machine (see :ref:`sftp_clients_label`)

#. Wait until you get the confirmation from the UniBE IT Service department that your access request is approved

#. After receiving the account activation confirmation, establish a VPN connection to the university network

#. Login to UBELIX with your SSH client via :bash:`[campusaccount]@submit.unibe.ch` to validate that the account is working

#. Configure your SFTP client for UBELIX

   -  File protocol: :bash:`SFTP`
   -  Port: :bash:`22`
   -  Host name: :bash:`submit.unibe.ch`
   -  User name: :bash:`[campusaccount]`
   -  Password: :bash:`[yoursecretpassword]`

.. tip::
    Check the documentation of your SSH and SFTP client to enable autologin
    (e.g. `Putty Autologin <https://superuser.com/a/44117>`_) or make configurations.


Project Handling
----------------
We expect you to work on your local machine and execute the experiments on UBELIX. To deploy your local code and its
changes to UBELIX we recommend you to use Github. If you setup your MIALab fork correctly you can update the code on
UBELIX by console without loosing information.

.. important::
   It's crucial that you work on your own fork of the MIALab repository! You need to fork the MIALab repository
   before proceeding with the next steps.

.. warning::
   Make sure that you do not add large-size files (>200kB, e.g. images, results) to your remote Github repository!
   Copy them manually from your local machine to UBELIX. For ignoring the appropriate folders / files modify your
   :bash:`.gitignore` file.


Clone Your Github Repository to UBELIX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This procedure needs to be performed once in order to clone the remote repository to UBELIX as a local repository.

.. important::
   Make sure that you do **not** clone the original MIALab repository (:bash:`https://github.com/ubern-mia/MIALab.git`)!
   Your remote repository URL should have the following pattern:
   :bash:`https://github.com/[yourgithubaccount]/MIALab.git`.

#. Login via SSH to UBELIX (:bash:`[campusaccount]@submit.unibe.ch`)

#. Create a new directory for the MIALab: :bash:`mkdir MIALab`

#. Change to your new directory: :bash:`cd MIALab`

#. Clone your remote repository: :bash:`git clone https://github.com/[yourgithubaccount]/MIALab.git`

#. Login via SFTP to UBELIX

#. Upload the images and additional large files (>200kB) manually to the correct directories on UBELIX


Update Your Local UBELIX Repository from Github
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This procedure needs to be performed when you want to update your code on UBELIX from Github.

.. important::
   Make sure that you commit and push your changes on your local machine to Github before updating the UBELIX
   repository.

#. Login via SSH to UBELIX (:bash:`[campusaccount]@submit.unibe.ch`)

#. Change to your MIALab base directory (e.g. :bash:`./MIALab`): :bash:`cd MIALab`

#. Update the local UBELIX repository from Github: :bash:`git pull origin master`

.. important::
   If you have multiple branches on Github modify the update command appropriately.


Setup Your UBELIX Python Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This procedure needs to be performed once before the first computation on UBELIX and after the cloning of your MIALab
fork onto UBELIX. For detailed information we refer to the
`official UBELIX Python documentation <https://hpc-unibe-ch.github.io/software/python.html>`_.

#. Login via SSH to UBELIX (:bash:`[campusaccount]@submit.unibe.ch`)

#. Load the Python module: :bash:`module load Anaconda3`

#. Prepare the environment for Python: :bash:`eval "$(conda shell.bash hook)"`

   -  This command needs to be executed after each :bash:`module load Anaconda3`
   -  Do **not** run :bash:`conda init` because it hardcodes environment variables and you need to rework the
      :bash:`.bashrc` file.

#. Create a new Python 3.7 environment with the name mialab (confirm with :bash:`y` when promoted during creation):
   :bash:`conda create -n mialab python=3.7`

#. Activate your new environment: :bash:`conda activate mialab`

#. Change to your MIALab base directory (e.g. :bash:`./MIALab`): :bash:`cd MIALab`

#. Install the dependencies of MIALab: :bash:`pip install -r requirements.txt`

.. important::
   If you require additional Python packages later in your project you can add them to your :bash:`requirements.txt`
   file and re-execute step 5 - 7 in the previous procedure.


Transfer Large-Size Data from UBELIX to your Local Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This procedure is typically used after an experiment is finished and when you need to analyze the results locally on
your machine.

#. Login via SFTP to UBELIX

#. Navigate to the appropriate directory

#. Copy the files to your local machine by drag-and-drop


Computation Job Handling
------------------------
The UBELIX contains a job scheduler (SLURM) to assign computational resources to jobs and to handle priorities.
The job scheduler is responsible that the necessary resources are available during the execution of the jobs and that
no aborts are generated due to unavailable resources.

All normally privileged users on UBELIX have exclusively access to the submission node (:bash:`submit.unibe.ch`) where
they submit their computational jobs via a job script. Writing a job script can be challenging at the beginning of
your HPC life. Therefore, we prepared a template job script for you below. If you need any further assistance, consult
the `official UBELIX documentation <https://hpc-unibe-ch.github.io/slurm/submission.html>`_ or ask a lecturer.


Writing A Job Script
^^^^^^^^^^^^^^^^^^^^
The job script specifies the resources you require for your computation. Because the experiments you will do in this
course require more or less similar resources we prepared a
`template job script <../additional_material/template_jobscript.sh>`_ for you.

.. code:: bash

   #!/bin/bash

   # SLURM Settings
   #SBATCH --job-name="GIVE_IT_A_NAME"
   #SBATCH --time=24:00:00
   #SBATCH --mem-per-cpu=128G
   #SBATCH --partition=epyc2
   #SBATCH --qos=job_epyc2
   #SBATCH --mail-user=your.name@students.unibe.ch
   #SBATCH --mail-type=ALL
   #SBATCH --output=%x_%j.out
   #SBATCH --error=%x_%j.err

   # Load Anaconda3
   module load Anaconda3
   eval "$(conda shell.bash hook)"

   # Load your environment
   conda activate mialab

   # Run your code
   srun python3 main_example_file.py

.. important::
   Do **not** use the GPU partition if you do not use specific libraries with GPU support! Your code does not magically
   speed-up when running on a GPU partition. Furthermore, MIALab as it is does not make use of GPUs!


Submitting & Controlling A Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The following procedure needs to be performed whenever you want to submit a computation job.

#. Write a job script or modify an existing one

#. Copy the job script to the correct location using the SFTP client

#. Submit the computation job by :bash:`sbatch [yourjobscriptname].sh`

.. important::
   Be aware of the paths inside the job script! Use relative paths from the location of the job script.


**Additional Useful Commands**

-  Monitor your jobs by :bash:`squeue --user [yourcampusaccont]`

-  Cancel one of your jobs by :bash:`scancel [yourjobid]`

.. important::
   Cancel jobs which contain errors such that other users can use the allocated resources.
