import os
import argparse
import glob
import shutil
import zipfile
import random

import SimpleITK as sitk
import numpy as np


def main(data_dir):

    previous_wd = os.getcwd()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(script_dir)

    out_train_dir = '../data/train/'
    if os.path.exists(out_train_dir):
        shutil.rmtree(out_train_dir)
    out_test_dir = '../data/test/'
    if os.path.exists(out_test_dir):
        shutil.rmtree(out_test_dir)

    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]

    print('unzip data')
    unzip_data_if_needed(data_dir)

    image_names, label_names = get_required_filenames()
    subject_files = get_files(data_dir, image_names, label_names)
    train_subjects, test_subjects = split_dataset(0.7, subject_files)

    image_transform = ComposeTransform([RescaleIntensity(),
                                        Resample((1., 1., 1.))])
    to_combine = {1: [2, 41, 7, 46],  # Grey Matter
                  2: [3, 42, 8, 47, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013,
                      1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029,
                      1030, 1031, 1032, 1033, 1034, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
                      2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026,
                      2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034],  # White Matter
                  3: [17, 53],  # Hippocampus
                  4: [18, 54],  # Amygdala
                  5: [10, 49]}  # Thalamus
    label_transform = ComposeTransform([Resample((1., 1., 1.)), MergeLabel(to_combine)])

    print('preparing training data')
    transform_and_write(train_subjects, image_transform, label_transform, out_train_dir)
    print('preparing testing data')
    transform_and_write(test_subjects, image_transform, label_transform, out_test_dir)

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


def get_required_filenames(native: bool = True, brain_mask: bool = False, bias_corr: bool = False):
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


def get_files(data_dir, image_names, label_names):

    def join_and_check_path(file_id, file_names):
        files = []
        for in_filename, out_filename in file_names:
            in_file_path = os.path.join(data_dir, file_id, in_filename)
            if not os.path.exists(in_file_path):
                raise ValueError('file "{}" not exists'.format(in_file_path))
            out_file_path = os.path.join(file_id, out_filename)
            files.append((in_file_path, out_file_path))
        return files

    subject_files = {}
    sub_dirs = glob.glob(data_dir + '/*')
    for sub_dir in sub_dirs:
        if not os.path.isdir(sub_dir):
            continue

        id_ = os.path.basename(sub_dir)

        image_files = join_and_check_path(id_, image_names)
        label_files = join_and_check_path(id_, label_names)
        subject_files[id_] = {'images': image_files, 'labels': label_files}
    return subject_files


def split_dataset(train_split, subject_files):
    seed = 20

    all_ids = list(subject_files.keys())
    random.Random(seed).shuffle(all_ids)

    n_train = int(len(all_ids)*train_split)
    train_ids = all_ids[:n_train]
    test_ids = all_ids[n_train:]

    train_subject = {k: subject_files[k] for k in train_ids}
    test_subject = {k: subject_files[k] for k in test_ids}

    return train_subject, test_subject


def transform_and_write(subject_files, image_transform, label_transform, out_dir):

    for id_, subject_file in subject_files.items():
        print(' - {}'.format(id_))

        for in_image_file, out_image_file in subject_file['images']:
            image = sitk.ReadImage(in_image_file, sitk.sitkUInt16)
            transformed_image = image_transform(image)
            out_image_path = os.path.join(out_dir, out_image_file)
            if not os.path.exists(os.path.dirname(out_image_path)):
                os.makedirs(os.path.dirname(out_image_path))
            sitk.WriteImage(transformed_image, out_image_path)

        for in_label_file, out_label_file in subject_file['labels']:
            label = sitk.ReadImage(in_label_file)
            transformed_label = label_transform(label)
            out_label_path = os.path.join(out_dir, out_label_file)
            if not os.path.exists(os.path.dirname(out_label_path)):
                os.makedirs(os.path.dirname(out_label_path))
            sitk.WriteImage(transformed_label, out_label_path)


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

    def __init__(self, min_=0, max_=65535) -> None:
        super().__init__()
        self.min = min_
        self.max = max_

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
        merged_img = np.zeros_like(np_img)

        for new_label, labels_to_merge in self.to_combine.items():
            indices = np.reshape(np.in1d(np_img.ravel(), labels_to_merge, assume_unique=True), np_img.shape)
            merged_img[indices] = new_label

        out_img = sitk.GetImageFromArray(merged_img)
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
