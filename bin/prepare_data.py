import os
import argparse
import glob
import shutil
import zipfile

import SimpleITK as sitk
import numpy as np


def main(data_dir):

    previous_wd = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    out_dir = '../data/train/'

    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    print('unzip data')
    unzip_data_if_needed(data_dir)

    print('preparing and copying data')
    image_names, label_names = get_required_filenames()
    subject_files = get_files(data_dir, out_dir, image_names, label_names)

    image_transform = ComposeTransform([RescaleIntensity(),
                                        Resample((1., 1., 1.))])
    to_combine = {1: [2, 41, 7, 46],  # Grey Matter
                  2: [3, 42, 8, 47, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034],  # White Matter
                  3: [17, 53],  # Hippocampus
                  4: [18, 54],  # Amygdala
                  5: [10, 49]}  # Thalamus
    label_transform = ComposeTransform([Resample((1., 1., 1.)), MergeLabel(to_combine)])
    transform_and_write(subject_files, image_transform, label_transform)

    os.chdir(previous_wd)

    print('done')


def unzip_data_if_needed(data_dir):
    zip_files = glob.glob(data_dir + '/*.zip')
    if len(zip_files) == 0:
        print('no files to unzip')

    for zip_file in zip_files:
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(path=os.path.dirname(zip_file))
        os.remove(zip_file)

    # clean up
    md5_files = glob.glob(data_dir + '/*.zip.md5')
    for md5_file in md5_files:
        os.remove(md5_file)


def get_required_filenames():
    # todo add as argument
    native = True
    brain_mask = False
    bias_corr = False

    images = []
    labels = []
    if native:
        images.append(('T1w/T1w_acpc_dc.nii.gz', 'T1native.nii.gz'))
        images.append(('T1w/T2w_acpc_dc.nii.gz', 'T2native.nii.gz'))
        if bias_corr:
            images.append(('T1w/T1w_acpc_dc_restore_brain.nii.gz', 'T1native_biasfieldcorr_noskull.nii.gz'))
            images.append(('T1w/T2w_acpc_dc_restore_brain.nii.gz', 'T2native_biasfieldcorr_noskull.nii.gz'))
        labels.append(('T1w/aparc+aseg.nii.gz', 'labels_native.nii.gz'))
        if brain_mask:
            labels.append(('T1w/brainmask_fs.nii.gz', 'Brainmasknative.nii.gz'))
    else:
        images.append(('MNINonLinear/T1w.nii.gz', 'T1mni.nii.gz'))
        images.append(('MNINonLinear/T2w.nii.gz', 'T2mni.nii.gz'))
        if bias_corr:
            images.append(('MNINonLinear/T1w_restore_brain.nii.gz', 'T1mni_biasfieldcorr_noskull.nii.gz'))
            images.append(('MNINonLinear/T2w_restore_brain.nii.gz', 'T2mni_biasfieldcorr_noskull.nii.gz'))
        labels.append(('MNINonLinear/aparc+aseg.nii.gz', 'labels_mniatlas.nii.gz'))
        if brain_mask:
            labels.append(('MNINonLinear/brainmask_fs.nii.gz', 'Brainmaskmni.nii.gz'))

    return tuple(images), tuple(labels)


def get_files(data_dir, out_dir, image_names, label_names):

    def join_and_check_path(id_, file_names):
        files = []
        for in_filename, out_filename in file_names:
            in_file_path = os.path.join(data_dir, id_, in_filename)
            if not os.path.exists(in_file_path):
                raise ValueError('file "{}" not exists'.format(in_file_path))
            out_file_path = os.path.join(out_dir, id_, out_filename)
            files.append((in_file_path, out_file_path))
        return files

    subject_files = {}
    subdirs = glob.glob(data_dir + '/*')
    for subdir in subdirs:
        if not os.path.isdir(subdir):
            continue

        id_ = os.path.basename(subdir)

        image_files = join_and_check_path(id_, image_names)
        label_files = join_and_check_path(id_, label_names)
        subject_files[id_] = {'images':image_files, 'labels': label_files}
    return subject_files


def transform_and_write(subject_files, image_tranform, label_transform):

    for id_, subject_file in subject_files.items():
        print(' - {}'.format(id_))

        for in_image_file, out_image_file in subject_file['images']:
            image = sitk.ReadImage(in_image_file, sitk.sitkUInt16)
            transformed_image = image_tranform(image)
            if not os.path.exists(os.path.dirname(out_image_file)):
                os.makedirs(os.path.dirname(out_image_file))
            sitk.WriteImage(transformed_image, out_image_file)

        for in_label_file, out_label_file in subject_file['labels']:
            label = sitk.ReadImage(in_label_file)
            transformed_label = label_transform(label)
            if not os.path.exists(os.path.dirname(out_label_file)):
                os.makedirs(os.path.dirname(out_label_file))
            sitk.WriteImage(transformed_label, out_label_file)


class Transform:
    def __call__(self, img: sitk.Image) -> sitk.Image:
        pass


class ComposeTransform(Transform):

    def __init__(self, transforms) -> None:
        super().__init__()
        self.transforms = transforms

    def __call__(self, img: sitk.Image) -> sitk.Image:
        for transform in self.transforms:
            img = transform(img)
        return img


class RescaleIntensity(Transform):

    def __init__(self, min=0, max=65535) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def __call__(self, img: sitk.Image) -> sitk.Image:
        return sitk.RescaleIntensity(img, self.min, self.max)


class Resample(Transform):

    def __init__(self, new_spacing: tuple) -> None:
        super().__init__()
        self.new_spacing = new_spacing

    def __call__(self, img: sitk.Image) -> sitk.Image:
        size, spacing, origin, direction = img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()

        scale = [ns / s for ns, s in zip(self.new_spacing, spacing)]
        new_size = [int(sz/sc) for sz, sc in zip(size, scale)]
        # new_origin = [o / sc for o, sc in zip(origin, scale)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        # resampler.SetTransform(sitk.Transform())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputDirection(direction)
        # resampler.SetOutputOrigin(new_origin)  # misfitted image when using adapted origin
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputSpacing(self.new_spacing)

        return resampler.Execute(img)


class MergeLabel(Transform):

    def __init__(self, to_combine: dict) -> None:
        super().__init__()
        # to_combine is a dict with keys -> new label and values -> list of labels to merge
        self.to_combine = to_combine

    def __call__(self, img: sitk.Image) -> sitk.Image:
        np_img = sitk.GetArrayFromImage(img)
        for new_label, labels_to_merge  in self.to_combine.items():
            np_img[np.in1d(np_img.ravel(), labels_to_merge, assume_unique=True).reshape(np_img.shape)] = new_label

        # set all non-selected labels to background
        all_labels = list(self.to_combine.keys())
        np_img[np.in1d(np_img.ravel(), all_labels, assume_unique=True, invert=True).reshape(np_img.shape)] = 0

        out_img = sitk.GetImageFromArray(np_img)
        out_img.CopyInformation(img)
        return out_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data preparation for the MIALab')
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='the path to the dataset'
    )

    args = parser.parse_args()
    main(args.data_dir)
