.. _post-processing_label:

Post-processing
===============

Can we leverage the segmentation performance by post-processing?

- Morphological operators
- Dense conditional random field (CRF)
- Manual user interaction (e.g., brushing)

Materials
^^^^^^^^^

- ``mialab.filtering.postprocessing``, e.g. use ``DenseCRF``
- \P. Krähenbühl and V. Koltun, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, Advances in Neural Information Processing Systems, vol. 24, pp. 109-117, 2011.
- \S. Nowozin and C. H. Lampert, Structured Learning and Prediction in Computer Vision, Foundations and Trends in Computer Graphics and Vision, vol. 6, pp. 185-365, 2010.
- \P. Cattin, Image Segmentation, 2016. [Online]. Available: https://www.miac.unibas.ch/SIP/pdf/SIP-07-Segmentation.pdf [Accessed: 08-Sep-2020], see chapter 6 - Mathematical Morphology

Evaluation
----------

Which metrics are suitable for our task? What is the influence of the validation procedure on the results?

- Metric types
- Influence of e.g. small structures
- Influence of validation procedure

Materials
^^^^^^^^^

- See ``pymia.evaluation`` package
- \A. A. Taha and A. Hanbury, Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool, BMC Med. Imaging, vol. 15, no. 1, pp. 1–28, 2015.
- `Cross-validation in machine learning <https://medium.com/towards-data-science/cross-validation-in-machine-learning-72924a69872f>`_
