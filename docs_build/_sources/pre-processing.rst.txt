.. _pre-processing_label:

Pre-processing
==============

Investigate the influence of pre-processing on the segmentation performance.

- Image smoothing / noise reduction
- Image normalization
- Histogram matching
- Skull stripping (separate the brain from the skull and other surrounding structures)

Materials
^^^^^^^^^

- ``pymia.filtering.preprocessing``
- `medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization <http://loli.github.io/medpy/generated/medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization.html#medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization>`_
- \L. G. Nyúl, J. K. Udupa, and X. Zhang, New variants of a method of MRI scale standardization, IEEE Trans. Med. Imaging, vol. 19, no. 2, pp. 143–50, Feb. 2000.
- J.-P. Bergeest and F. Jäger, A Comparison of Five Methods for Signal Intensity Standardization in MRI, in Bildverarbeitung für die Medizin 2008, Berlin Heidelberg: Springer, 2008, pp. 36–40.
- `scikit-learn: Pre-processing data <http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing>`_
