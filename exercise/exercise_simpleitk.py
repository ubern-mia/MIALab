
import sys
import os

import numpy as np
import SimpleITK as sitk

try:
    import exercise.helper as helper
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), r'C:\Users\stude\OneDrive\Master\3.Semester\Medical Image Analysis Lab\Code\MIALab_Lukas_Studer'))
    import exercise.helper as helper


def load_image(img_path, is_label_img):
    # todo: load the image from the image path with the SimpleITK interface (hint: 'ReadImage')
    # todo: if 'is_label_img' is True use argument outputPixelType=sitk.sitkUInt8,
    #  else use outputPixelType=sitk.sitkFloat32

    pixel_type = None  # todo: modify here
    img = None  # todo: modify here

    return img


def to_numpy_array(img):
    # todo: transform the SimpleITK image to a numpy ndarray (hint: 'GetArrayFromImage')
    np_img = None  # todo: modify here

    return np_img


def to_sitk_image(np_image, reference_img):
    # todo: transform the numpy ndarray to a SimpleITK image (hint: 'GetImageFromArray')
    # todo: do not forget to copy meta-information (e.g., spacing, origin, etc.) from the reference image
    #  (hint: 'CopyInformation')! (otherwise defaults are set)

    img = None  # todo: modify here
    # todo: ...

    return img


def register_images(img, label_img, atlas_img):
    registration_method = _get_registration_method(atlas_img, img)  # type: sitk.ImageRegistrationMethod
    # todo: execute the registration_method to the img (hint: fixed=atlas_img, moving=img)
    # the registration returns the transformation of the moving image (parameter img) to the space of
    # the atlas image (atlas_img)
    transform = None  # todo: modify here

    # todo: apply the obtained transform to register the image (img) to the atlas image (atlas_img)
    # hint: 'Resample' (with referenceImage=atlas_img, transform=transform, interpolator=sitkLinear,
    # defaultPixelValue=0.0, outputPixelType=img.GetPixelIDValue())
    registered_img = None  # todo: modify here

    # todo: apply the obtained transform to register the label image (label_img) to the atlas image (atlas_img), too
    # be careful with the interpolator type for label images!
    # hint: 'Resample' (with interpolator=sitkNearestNeighbor, defaultPixelValue=0.0,
    # outputPixelType=label_img.GetPixelIDValue())
    registered_label = None  # todo: modify here

    return registered_img, registered_label


def preprocess_rescale_numpy(np_img, new_min_val, new_max_val):
    max_val = np_img.max()
    min_val = np_img.min()
    # todo: rescale the intensities of the np_img to the range [new_min_val, new_max_val]. Use numpy arithmetics only.
    rescaled_np_img = None  # todo: modify here

    return rescaled_np_img


def preprocess_rescale_sitk(img, new_min_val, new_max_val):
    # todo: rescale the intensities of the img to the range [new_min_val, new_max_val] (hint: RescaleIntensity)
    rescaled_img = None  # todo: modify here

    return rescaled_img


def extract_feature_median(img):
    # todo: apply median filter to image (hint: 'Median')
    median_img = None  # todo: modify here

    return median_img


def postprocess_largest_component(label_img):
    # todo: get the connected components from the label_img (hint: 'ConnectedComponent')
    connected_components = None  # todo: modify here

    # todo: order the component by ascending component size (hint: 'RelabelComponent')
    relabeled_components = None  # todo: modify here

    largest_component = relabeled_components == 1  # zero is background
    return largest_component


# --- DO NOT CHANGE
def _get_registration_method(atlas_img, img) -> sitk.ImageRegistrationMethod:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.2)

    registration_method.SetMetricUseFixedImageGradientFilter(False)
    registration_method.SetMetricUseMovingImageGradientFilter(False)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(atlas_img, img,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    return registration_method


# --- DO NOT CHANGE
if __name__ == '__main__':
    callback = helper.TestCallback()
    callback.start('SimpleITK')

    callback.start_test('load_image')
    img_ = load_image('../data/exercise/subjectX/T1native.nii.gz', False)
    load_ok = all((isinstance(img_, sitk.Image),
                   img_.GetPixelID() == 8,
                   img_.GetSize() == (181, 217, 181),
                   img_.GetPixel(100, 100, 100) == 12175,
                   img_.GetPixel(100, 100, 101) == 11972))
    callback.end_test(load_ok)

    callback.start_test('to_numpy_array')
    np_img_ = to_numpy_array(img_)
    to_numpy_ok = all((isinstance(np_img_, np.ndarray),
                       np_img_.dtype.name == 'float32',
                       np_img_.shape == (181, 217, 181),
                       np_img_[100, 100, 100] == 12175,
                       np_img_[101, 100, 100] == 11972))
    callback.end_test(to_numpy_ok)

    callback.start_test('to_sitk_image')
    rev_img_ = to_sitk_image(np_img_, img_)
    to_sitk_ok = all((isinstance(rev_img_, sitk.Image),
                      rev_img_.GetOrigin() == img_.GetOrigin(),
                      rev_img_.GetSpacing() == img_.GetSpacing(),
                      rev_img_.GetDirection() == img_.GetDirection(),
                      rev_img_.GetPixel(100, 100, 100) == 12175,
                      rev_img_.GetPixel(100, 100, 101) == 11972))
    callback.end_test(to_sitk_ok)

    callback.start_test('register_images')
    atlas_img_ = load_image('../data/exercise/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz', False)
    label_img_ = load_image('../data/exercise/subjectX/labels_native.nii.gz', True)
    if isinstance(atlas_img_, sitk.Image) and isinstance(label_img_, sitk.Image):
        registered_img_, registered_label_ = register_images(img_, label_img_, atlas_img_)
        if isinstance(registered_img_, sitk.Image) and isinstance(registered_label_, sitk.Image):
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(registered_img_, registered_label_)
            labels = tuple(sorted(stats.GetLabels()))
            register_ok = all((registered_img_.GetSize() == registered_label_.GetSize() == (197, 233, 189),
                               labels == tuple(range(6))))
        else:
            register_ok = False
    else:
        register_ok = False
    callback.end_test(register_ok)

    callback.start_test('preprocss_rescale_numpy')
    if isinstance(np_img_, np.ndarray):
        pre_np = preprocess_rescale_numpy(np_img_, -3, 101)
        if isinstance(pre_np, np.ndarray):
            pre_np_ok = np.min(pre_np) == -3 and np.max(pre_np) == 101
        else:
            pre_np_ok = False
    else:
        pre_np_ok = False
    callback.end_test(pre_np_ok)

    callback.start_test('preprocss_rescale_sitk')
    pre_sitk = preprocess_rescale_sitk(img_, -3, 101)
    if isinstance(pre_sitk, sitk.Image):
        min_max = sitk.MinimumMaximumImageFilter()
        min_max.Execute(pre_sitk)
        pre_sitk_ok = min_max.GetMinimum() == -3 and min_max.GetMaximum() == 101
    else:
        pre_sitk_ok = False
    callback.end_test(pre_sitk_ok)

    callback.start_test('extract_feature_median')
    median_img_ = extract_feature_median(img_)
    if isinstance(median_img_, sitk.Image):
        median_ref = load_image('../data/exercise/subjectX/T1med.nii.gz', False)
        if isinstance(median_ref, sitk.Image):
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(median_img_ - median_ref)
            median_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
        else:
            median_ok = False
    else:
        median_ok = False
    callback.end_test(median_ok)

    callback.start_test('postprocess_largest_component')
    largest_hippocampus = postprocess_largest_component(label_img_ == 3)  # 3: hippocampus
    if isinstance(largest_hippocampus, sitk.Image):
        largest_ref = load_image('../data/exercise/subjectX/hippocampus_largest.nii.gz', True)
        if isinstance(largest_ref, sitk.Image):
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(largest_hippocampus - largest_ref)
            post_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
        else:
            post_ok = False
    else:
        post_ok = False
    callback.end_test(post_ok)

    callback.end()
