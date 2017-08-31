# Medical Image Analysis Laboratory 2017

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

During the laboratory you will team up in groups of 2-3 students and investigate one of these pipeline elements in-depth (see [experiments](docs/experiments.md)).
You will get to know and to use various libraries and software tools needed in the daily life as biomedical engineer in the medical image analysis domain (see [tools](docs/tools.md)).
Note the [syllabus](docs/syllabus.rst) for a rough plan of the next 14 weeks and deadlines.

Enjoy!

----

Found a bug or do you have suggestions? Open an issue or better submit a pull request.

## Data

###### Medical Images

The medical images will be provided trough Ilias.

The dataset consists of 3T head MRIs of 100 unrelated subjects from the [Human Connectome Project](https://www.humanconnectome.org/) (HCP) dataset of healthy volunteers [2]. For each subject, the following data is available:

* T1-weighted MR image volume, skull-stripped and non-skull-stripped (but defaced for anonymization [3])
* T2-weighted MR image volume, processed the same way as the T1 image
* Transform file from subject to MNI-atlas space [4]
* White matter ground truth labels:
  * 1: cortical white matter
  * 2: left hemisphere cortical white matter
  * 3: right hemisphere cortical white matter
* FreeSurfer-generated brain mask label volumes

The ground truth labels are generated by FreeSurfer [e.g. 5]  and are not manual expert annotations.

###### Toy Example

The toy example data in the data directory is taken from the Sherwood library [1].

## References

[1] Microsoft Research, "Sherwood C++ and C# code library for decision forests", 2012. [Online]. Available: http://research.microsoft.com/en-us/downloads/52d5b9c3-a638-42a1-94a5-d549e2251728/. [Accessed: 16-Mar-2016].

[2] Van Essen, D.C., Smith, S.M., Barch, D.M., Behrens, T.E., Yacoub, E., Ugurbil, K. and Wu-Minn HCP Consortium, 2013. [The WU-Minn human connectome project: an overview](http://www.sciencedirect.com/science/article/pii/S1053811913005351). *Neuroimage*, *80*, pp.62-79. 

[3] Milchenko, M. and Marcus, D., 2013. [Obscuring surface anatomy in volumetric imaging data](https://link.springer.com/article/10.1007/s12021-012-9160-3). *Neuroinformatics*, *11*(1), pp.65-75.

[4] Mazziotta, J., Toga, A., Evans, A., Fox, P., Lancaster, J., Zilles, K., Woods, R., Paus, T., Simpson, G., Pike, B. and Holmes, C., 2001. [A probabilistic atlas and reference system for the human brain: International Consortium for Brain Mapping (ICBM)](http://rstb.royalsocietypublishing.org/content/356/1412/1293.short). *Philosophical Transactions of the Royal Society of London B: Biological Sciences*, *356*(1412), pp.1293-1322.

[5] Fischl, B., Salat, D.H., Busa, E., Albert, M., Dieterich, M., Haselgrove, C., Van Der Kouwe, A., Killiany, R., Kennedy, D., Klaveness, S. and Montillo, A., 2002. [Whole brain segmentation: automated labeling of neuroanatomical structures in the human brain](http://www.sciencedirect.com/science/article/pii/S089662730200569X). Neuron, 33(3), pp.341-355.

