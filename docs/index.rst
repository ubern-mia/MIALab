.. MIALab documentation master file, created by
   sphinx-quickstart on Thur August 31 10:58:00 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
Medical Image Analysis Laboratory
==================================

Welcome to the medical image analysis laboratory (MIALab) 2017.
This repository contains all code you will need to get started with medical image analysis.

During the MIALab you will work on the task of brain tumor segmentation from MR images.
We have set up an entire pipeline to solve this task, specifically:

- Pre-processing
- Registration
- Feature extraction
- Voxel-wise tissue classification using a decision forest
- Post-processing
- Evaluation

During the laboratory you will team up in groups of 2-3 students and investigate one of these pipeline elements in-depth (see experiments).
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer in the medical image analysis domain (see tools).
Note the syllabus for a rough plan of the next 14 weeks and deadlines.

Enjoy!


.. toctree::
    :maxdepth: 1
    :caption: Getting started

    installation
    syllabus
    experiments
    data
    tools

.. toctree::
    :maxdepth: 2
    :caption: Packages

    package_classifier
    package_data
    package_evaluation
    package_filtering

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
