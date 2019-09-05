.. _pipeline_label:

Pipeline
========

The figure below shows our medical image analysis (MIA) pipeline with its single steps. Our pipeline has as input two magnetic resonance (MR) image slices (i.e., a T1-weighted (T1w) image slice and a T2-weighted (T2w) image slice) and a segmentation of the brain into the structures described previously (see :ref:`background_label`).
The pipeline itself consists of the following steps:

- Registration, which aims at aligning the two MR images
- Pre-processing, which aims at improving the image quality for our machine learning algorithm
- Feature extraction, which aims to extract meaningful features from the MR images for the subsequent classification
- Classification, which performs a voxel-wise tissue classification using the extracted features
- Post-processing, which aims to improve the classification

The dashed boxes indicate pre-steps or selections that influence a step. The provided experiments (see :ref:`experiments_label`) correspond to boxes in the figure. Additionally, we will also have a look at the evaluation of such a pipeline.

.. image:: pics/pipeline.png
   :width: 600

An in-depth description of the concept of the pipeline with references for further reading can be found in [1]_.

References
----------

.. [1] Pereira, S., Pinto, A., Oliveira, J., Mendrik, A. M., Correia, J. H., Silva, C. A.: Automatic brain tissue segmentation in MR images using Random Forests and Conditional Random Fields. Journal of Neuroscience Methods 270, 111-123, (2016). https://doi.org/10.1016/j.jneumeth.2016.06.017
