import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))  # append the MIALab root directory to Python path

import SimpleITK as sitk
import pymia.data.loading as load
import pymia.filtering.filter as fltr
import pymia.filtering.registration as fltr_reg

import mialab.data.structure as structure
import mialab.utilities.file_access_utilities as futil
import exercise.helper as helper


def collect_image_paths(data_dir):
    image_keys = [structure.BrainImageTypes.T1,
                  structure.BrainImageTypes.GroundTruth]

    class MyFilePathGenerator(load.FilePathGenerator):
        @staticmethod
        def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
            if file_key == structure.BrainImageTypes.T1:
                file_name = 'T1native'
            elif file_key == structure.BrainImageTypes.GroundTruth:
                file_name = 'labels_native'
            else:
                raise ValueError('Unknown key')
            return os.path.join(root_dir, file_name + file_extension)

    dir_filter = futil.DataDirectoryFilter()

    # todo: create an instance of load.FileSystemDataCrawler and pass the correpsonding arguments
    crawler = None  # todo: modify here

    return crawler


def load_images(image_paths):
    # todo: read the images (T1 as sitk.sitkFloat32, GroundTruth as sitk.sitkUInt8)
    image_dict = {
        structure.BrainImageTypes.T1: None,  # todo: modify here
        structure.BrainImageTypes.GroundTruth: None  # todo: modify here
    }

    return image_dict


def register_images(image_dict, atlas_img):

    registration = fltr_reg.MultiModalRegistration()
    registration_params = fltr_reg.MultiModalRegistrationParams(atlas_img)
    # todo execute the registration with the T1-weighted image and the registration parameters
    registered_t1 = None  # todo: modify here

    gt_img = image_dict[structure.BrainImageTypes.GroundTruth]
    # todo: apply transform to GroundTruth image (gt_img) (hint: sitk.Resample, referenceImage=atlas_img, transform=tegistration.transform, interpolator=sitk.sitkNearestNeighbor
    registered_gt = None  # todo: modify here

    return registered_t1, registered_gt


def preprocess_filter_rescale_t1(image_dict, new_min_val, new_max_val):
    class MinMaxRescaleFilterParams(fltr.IFilterParams):
        def __init__(self, min_, max_) -> None:
            super().__init__()
            self.min = min_
            self.max = max_

    class MinMaxRescaleFilter(fltr.IFilter):
        def execute(self, img: sitk.Image, params: MinMaxRescaleFilterParams = None) -> sitk.Image:
            resacaled_img = sitk.RescaleIntensity(img, params.min, params.max)
            return resacaled_img

    # todo: use the above filter and parameters to get the rescaled T1-weighted image
    filter = None  # todo: modify here
    filter_params = None  # todo: modify here
    minmax_rescaled_img = None  # todo: modify here

    return minmax_rescaled_img


def extract_feature_median_t1(image_dict):

    class MedianFilter(fltr.IFilter):
        def execute(self, img: sitk.Image, params: fltr.IFilterParams = None) -> sitk.Image:
            med_img = sitk.Median(img)
            return med_img

    # todo: use the above filter class to get the median image feature of the T1-weighted image
    filter = None  # todo: modify here
    median_img = None  # todo: modify here

    return median_img


# --- DO NOT CHANGE
if __name__ == '__main__':
    callback = helper.TestCallback()
    callback.start('Pipeline')

    callback.start_test('collect_image_paths')
    crawler = collect_image_paths('../data/exercise/')
    if isinstance(crawler, load.FileSystemDataCrawler):
        image_paths = crawler.data
        subjectx_paths = image_paths.get('subjectX')  # only consider subjectX
        identifier = subjectx_paths.pop('subjectX', '')
        collect_ok = identifier.endswith('subjectX') and structure.BrainImageTypes.GroundTruth in subjectx_paths \
                     and structure.BrainImageTypes.T1 in subjectx_paths
    else:
        collect_ok = False
        subjectx_paths = None  # for load_images
    callback.end_test(collect_ok)

    callback.start_test('load_images')
    if isinstance(subjectx_paths, dict):
        subjectx_images = load_images(subjectx_paths)
        load_ok = isinstance(subjectx_images, dict) and all(isinstance(img, sitk.Image) for img in subjectx_images.values())
    else:
        load_ok = False
        subjectx_images = None  # for preprocess_filter_rescale_t1
    callback.end_test(load_ok)

    callback.start_test('register_images')
    atlas_img = sitk.ReadImage('../data/exercise/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz')
    if isinstance(subjectx_paths, dict):
        registered_img, registered_gt = register_images(subjectx_images, atlas_img)
        if isinstance(registered_img, sitk.Image) and isinstance(registered_gt, sitk.Image):
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(registered_img, registered_gt)
            labels = stats.GetLabels()
            register_ok = registered_img.GetSize() == registered_gt.GetSize() == (197, 233, 189) and labels == tuple(range(6))
        else:
            register_ok = False
    else:
        register_ok = False
    callback.end_test(register_ok)

    callback.start_test('preprocess_filter_rescale_t1')
    if isinstance(subjectx_images, dict):
        pre_rescale = preprocess_filter_rescale_t1(subjectx_images, -3, 101)
        if isinstance(pre_rescale, sitk.Image):
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(pre_rescale)
            pre_ok = min_max.GetMinimum() == -3 and min_max.GetMaximum() == 101
        else:
            pre_ok = False
    else:
        pre_ok = False
    callback.end_test(pre_ok)

    callback.start_test('extract_feature_median_t1')
    if isinstance(subjectx_images, dict):
        median_img = extract_feature_median_t1(subjectx_images)
        if isinstance(median_img, sitk.Image):
            median_ref = sitk.ReadImage('../data/exercise/subjectX/T1med.nii.gz')
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(median_img - median_ref)
            median_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
        else:
            median_ok = False
    else:
        median_ok = False
    callback.end_test(median_ok)

    callback.end()
