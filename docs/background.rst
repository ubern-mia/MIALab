.. _background_label:

Clinical Background
===================

In the MIALab, we are segmenting structures of the brain. We are thus focusing on the most prominent medical imaging analysis (MIA) task, segmentation, and do it in the most prominent area in MIA, the brain, on magnetic resonance imaging (MRI) images.
Segmenting brain structures from MRI images is important, e.g., for the tracking of progression in neurodegenerative diseases by the atrophy of brain tissue [1]_. Performing the segmentation task manually is very time-consuming, user-dependent, and costly [2]_.
This is why we aim for an automated approach based on machine learning (ML).

The aim of the pipeline is to classify each voxel of a brain MRI in one of the following classes:

- 0: Background (or any other structures than the one listed below)
- 1: Cortical and cerebellar white matter
- 2: Cerebral and cerebellar cortex / grey matter
- 3: Hippocampus
- 4: Amygdala
- 5: Thalamus

An example sagittal image slice is shown in the figure below, where the label image (reference segmentation referred to as ground truth or simply labels) is shown next to the two available MRI sequences (T1-weighted and T2-weigthed).

.. image:: pics/background.png
   :width: 600

References
----------

.. [1] Pereira, S., Pinto, A., Oliveira, J., Mendrik, A. M., Correia, J. H., Silva, C. A.: Automatic brain tissue segmentation in MR images using Random Forests and Conditional Random Fields. Journal of Neuroscience Methods 270, 111-123, (2016). https://doi.org/10.1016/j.jneumeth.2016.06.017

.. [2] Porz, N., Bauer, S., Pica, A., Schucht, P., Beck, J., Verma, R.K., Slotboom, J., Reyes, M., Wiest, R.: Multi-Modal Glioblastoma Segmentation: Man versus Machine. PLoS ONE 9(5), (2014). https://doi.org/10.1371/journal.pone.0096873