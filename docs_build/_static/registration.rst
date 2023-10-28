.. _registration_label:

Registration
============

What is the optimal setting to register the images to an atlas?

- Transformation type
- Metric type
- Optimizer type
- Deep learning for image registration

Materials
^^^^^^^^^

- ``pymia.filtering.registration``
- \P. Viola and W. M. I. Wells, Alignment by maximization of mutual information, Proc. IEEE Int. Conf. Comput. Vis., vol. 24, no. 2, pp. 16–23, 1995.
- \P. Cattin and V. Roth, Biomedical Image Analysis, 2016. [Online]. Available: https://miac.unibas.ch/BIA/ [Accessed: 08-Sep-2020].
- M.-M. Rohé, M. Datar, T. Heimann, M. Sermesant, and X. Pennec, “SVF-Net: Learning Deformable Image Registration Using Shape Matching,” in Medical Image Computing and Computer Assisted Intervention − MICCAI 2017: 20th International Conference, Quebec City, QC, Canada, September 11-13, 2017, Proceedings, Part I, Springer International Publishing, 2017, pp. 266–274.
- `SimpleITK Notebooks <http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/>`_: See chapters 60-67
- `ITK Software Guide, Book 2 <https://itk.org/ITKSoftwareGuide/html/Book2/ITKSoftwareGuide-Book2ch3.html>`_: In C++ but with a thorough description

Tools
^^^^^
There exist various tools for registration besides the implemented code for registration:

- `3D Slicer <https://www.slicer.org/>`_: Open source software which also includes registration.
- `ANTs <http://stnava.github.io/ANTs/>`_: Advanced Normalization Tools, which come with registration algorithms.
- `NiftyReg <http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg>`_: Rigid, affine and non-linear registration of medical images.
- `SimpleElastix <https://simpleelastix.github.io/>`_: An extension of SimpleITK.
