========================================
Using UBELIX, the HPC cluster of the university
========================================

.. role:: bash(code)
   :language: bash

UBELIX is the HPC cluster of the university of Bern, which you can use after activating your campus account. Besides this short guide, we recommend reading the `official documentation <https://docs.id.unibe.ch/ubelix>`_.

#. Request access as described `here <https://docs.id.unibe.ch/ubelix/ubelix-101/account-activation>`_.
#. As soon as you get access, you can `login <https://docs.id.unibe.ch/ubelix/ubelix-101/login-to-the-cluster>`_ (if you are not in the university network, you have to use `VPN <http://www.unibe.ch/university/campus_and_infrastructure/rund_um_computer/internetzugang/access_to_internal_resources_via_vpn/index_eng.html>`_).
#. Create a directory for the MIA Lab, e.g.: :bash:`mkdir MIALab2017`
#. Change to this directory: :bash:`cd MIALab2017`
#. Clone the git repository (replace with your forked repo): :bash:`git clone https://github.com/istb-mia/MIALab.git`
#. Get the data on UBELIX:
	- First, change to the data directory: :bash:`cd data`
	- Get the atlas files :bash:`wget https://www.dropbox.com/s/k6l4d9hnlw9mold/atlas.zip`
	- Get the training data :bash:`wget https://www.dropbox.com/s/649mlwgjjwwebma/train_data.zip`
	- Get the test data :bash:`wget https://www.dropbox.com/s/r2n3ctow29ehgw4/test_data.zip`
	- Unzip: :bash:`unzip train_data.zip` :bash:`unzip trest_data.zip` :bash:`unzip atlas.zip`
	- Remove zip files to save space :bash:`rm *.zip`
	
Now, we're almost ready to go. As in our local installation, we have to get miniconda / anaconda and create a python environment
#. Install miniconda and create a mialab environment (see the installation page)
#. Activate the mialab environment: :bash:`activate mialab`
#. Install tensorflow with :bash:`conda install tensorflow==1.2.1`
#. If you're still in the data folder, move to the localtion where the file :bash:`setup.py` is: :bash:`cd ..`
#. Run the installation with :bash:`pip install .`

Please note that the 'hello world' example will fail because matplotlib cannot open a window to display the plot (no gui...).

Now you're ready to go! To submit a job to the cluster, you need to create a small script, e.g. :bash:`jobsript.sh`

	
.. code-block:: bash

		#!/bin/bash

		#SBATCH --mail-user=youremail@students.unibe.ch
		#SBATCH --mail-type=begin,end,fail
		#SBATCH --time=01:15:00
		#SBATCH --mem-per-cpu=10G
		#SBATCH --cpus-per-task=8
		#SBATCH --mem-per-cpu=10G
		#SBATCH --workdir=.
		#SBATCH --job-name="lab_test"

		#### Your shell commands below this line ####
		export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

		# folders
		workdir=${PWD}

		# activate environment
		source activate mialab
		python -V

		python <PATH TO YOUR MIALAB INSTALLATION>/bin/main.py

Adapt the cpus-per-task, mem-per-cpu, and the time as required for your experiment.

After creating/uploading the jobscript, you have to make in executable with :bash:`chmod +x jobscript.sh`

#. Submit the job with :bash:`sbatch jobscript.sh`
#. Check the status of your job with :bash:`squeue -u <YOUR USERNAME>`
#. Enjoy :-)
