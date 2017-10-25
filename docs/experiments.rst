===========
Experiments
===========

You will form groups of 2-3 people and select one topic to investigate and experiment.
A list of materials (papers, lectures, links to Python libraries, and code reference) is given for each experiment as starting point.

Pre-processing
---------------

Investigate the influence of pre-processing on the segmentation performance.

- Image smoothing / noise reduction
- Image normalization
- Histogram matching
- Skull stripping (separate the brain from the skull and other surrounding structures)

Materials
^^^^^^^^^

- ``mialab.filtering.preprocessing``, e.g. ``SkullStrip``
- `medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization <http://loli.github.io/medpy/generated/medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization.html#medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization>`_
- \L. G. Nyúl, J. K. Udupa, and X. Zhang, New variants of a method of MRI scale standardization, IEEE Trans. Med. Imaging, vol. 19, no. 2, pp. 143–50, Feb. 2000.
- J.-P. Bergeest and F. Jäger, A Comparison of Five Methods for Signal Intensity Standardization in MRI, in Bildverarbeitung für die Medizin 2008, Berlin Heidelberg: Springer, 2008, pp. 36–40.
- `scikit-learn: Pre-processing data <http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing>`_

Registration
-------------

What is the optimal setting to register the images to an atlas?

- Transformation type
- Metric type
- Optimizer type
- Deep learning for image registration

Materials
^^^^^^^^^

- ``mialab.filtering.registration``, e.g. use the ``RegistrationPlotter``
- \P. Viola and W. M. I. Wells, Alignment by maximization of mutual information, Proc. IEEE Int. Conf. Comput. Vis., vol. 24, no. 2, pp. 16–23, 1995.
- \P. Cattin and V. Roth, Biomedical Image Analysis, 2016. [Online]. Available: https://miac.unibas.ch/BIA/ [Accessed: 25-Aug-2017].
- M.-M. Rohé, M. Datar, T. Heimann, M. Sermesant, and X. Pennec, “SVF-Net: Learning Deformable Image Registration Using Shape Matching,” in Medical Image Computing and Computer Assisted Intervention − MICCAI 2017: 20th International Conference, Quebec City, QC, Canada, September 11-13, 2017, Proceedings, Part I, Springer International Publishing, 2017, pp. 266–274.
- `SimpleITK Notebooks <http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/>`_: See chapters 60-67
- `ITK Software Guide, Book 2 <https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html>`_: In C++ but with a thorough description

Post-processing
----------------

Can we leverage the segmentation performance by post-processing?

- Morphological operators
- Dense conditional random field (CRF)
- Manual user interaction (e.g., brushing)

Materials
^^^^^^^^^

- ``mialab.filtering.postprocessing``, e.g. use ``DenseCRF``
- \P. Krähenbühl and V. Koltun, Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials, Advances in Neural Information Processing Systems, vol. 24, pp. 109-117, 2011.
- \S. Nowozin and C. H. Lampert, Structured Learning and Prediction in Computer Vision, Foundations and Trends in Computer Graphics and Vision, vol. 6, pp. 185-365, 2010.
- \P. Cattin and V. Roth, Biomedical Image Analysis - Mathematical Morphology, 2013. [Online]. Available: http://informatik.unibas.ch/fileadmin/Lectures/FS2013/CS252/morphology13.pdf [Accessed: 25-Aug-2017]

Evaluation
-----------

Which metrics are suitable for our task? What is the influence of the validation procedure on the results?

- Metric types
- Influence of e.g. small structures
- Influence of validation procedure

Materials
^^^^^^^^^

- ``mialab.evaluation.metrics``
- \A. A. Taha and A. Hanbury, Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool, BMC Med. Imaging, vol. 15, no. 1, pp. 1–28, 2015.
- `Cross-validation in machine learning <https://medium.com/towards-data-science/cross-validation-in-machine-learning-72924a69872f>`_

Machine Learning Algorithms
----------------------------

Do other machine learning algorithms perform better on our task? Can we improve the segmentation performance by parameter tuning?

- Overfitting
- Parameter tuning (tree depth, forest size)
- Support Vector Machine (SVM)
- Variants of decision forests (e.g., gradient boosted trees)

Materials
^^^^^^^^^

- `scikit-learn: Supervised machine learning algorithms <http://scikit-learn.org/stable/supervised_learning.html#supervised-learning>`_
- \A. Criminisi and J. Shotton, Decision Forests for Computer Vision and Medical Image Analysis, 1st ed. London: Springer, 2013.
- \R. S. Olson, W. La Cava, Z. Mustahsan, A. Varik, and J. H. Moore, Data-driven Advice for Applying Machine Learning to Bioinformatics Problems, Aug. 2017.

Feature Engineering
--------------------

What features could be used to improve our model?

- Investigate other features

  - Hemisphere feature
  - Filter banks
  - Histogram of oriented gradients (HOGs)

- 2-D / 3-D differences

Materials
^^^^^^^^^

- `scikit-image feature module <http://scikit-image.org/docs/dev/api/skimage.feature.html>`_

Feature Selection
------------------

Can we reduce the number of features to decrease the model complexity and the computational burden.

- Decision forest feature importance
- Principal component analysis (PCA)
- Mutual information based feature selection

Materials
^^^^^^^^^

- `scikit-learn: Dimensionality reduction <http://scikit-learn.org/stable/modules/decomposition.html#decompositions>`_
- `Parallelized Mutual Information based Feature Selection <https://github.com/danielhomola/mifs>`_
- \H. Peng, F. Long, and C. Ding, Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 27, no. 8, pp. 1226-38, 2005.

Deep Learning
--------------

Deep learning has gained much attention in the last years outperforming methods such as decision forests. What is the performance of a deep learning method on our task?

- Implement a deep learning method

Materials
^^^^^^^^^

- `Generic U-Net Tensorflow implementation for image segmentation <https://github.com/jakeret/tf_unet>`_
- \O. Ronneberger, P. Fischer, and T. Brox, U-Net: Convolutional Networks for Biomedical Image Segmentation, May 2015.