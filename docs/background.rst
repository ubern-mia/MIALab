Clinical Background
===================

In this laboratory, we are segmenting structures of the brain. We are thus focusing on the most prominent MIA task: segmentation, and do it in the most prominent area in MIA: the brain.
Segementing brain structures is important for the tracking of progression in nerodegenerative diseases by the atrophy of brain tissue. Performing the segmentation task manually is very time-consuming.
This is why we aim for an automated approach based on machine learning (ML).


The aim of the pipeline is to classify each voxel of a brain MRI in one of the following classes:
- 0: Background (or other structures)
- 1: Cortical and cerebellar white matter
- 2: Cerebral and cerebellar cortex / grey matter
- 3: Hippocampus
- 4: Amygdala
- 5: Thalamus

An example case is shown in the figure below, where the segmentation is shown next to the two available sequences (T1-weighted and T2-weigthed).

.. image:: pics/background.png
   :width: 600

