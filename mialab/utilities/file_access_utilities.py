"""This modules contains utility functions and classes for the access of the file system."""
import os
from typing import List

import mialab.data.loading as load
import mialab.data.structure as structure


class BrainImageFilePathGenerator(load.FilePathGenerator):
    """Represents a brain image file path generator.

    The generator is used to convert a human readable image identifier to an image file path,
    which allows to load the image.
    """

    def __init__(self):
        """Initializes a new instance of the BrainImageFilePathGenerator class."""
        pass

    @staticmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """Gets the full file path for an image.

        Args:
            id_ (str): The image identification.
            root_dir (str): The image' root directory.
            file_key (object): A human readable identifier used to identify the image.
            file_extension (str): The image' file extension.

        Returns:
            str: The images' full file path.
        """

        # the commented file_names are for the registration group

        if file_key == structure.BrainImageTypes.T1:
            # file_name = 'T1native'
            file_name = 'T1mni_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.T2:
            # file_name = 'T2native'
            file_name = 'T2mni_biasfieldcorr_noskull'
        elif file_key == structure.BrainImageTypes.GroundTruth:
            # file_name = 'labels_native'
            file_name = 'labels_mniatlas'
        elif file_key == structure.BrainImageTypes.BrainMask:
            # file_name = 'Brainmasknative'
            file_name = 'Brainmaskmni'
        else:
            raise ValueError('Unknown key')

        return os.path.join(root_dir, file_name + file_extension)


class DataDirectoryFilter(load.DirectoryFilter):
    """Represents a data directory filter.

    The filter is used to
    """

    def __init__(self):
        """Initializes a new instance of the DataDirectoryFilter class."""
        pass

    @staticmethod
    def filter_directories(dirs: List[str]) -> List[str]:
        """Filters a list of directories.

        Args:
            dirs (List[str]): A list of directories.

        Returns:
            List[str]: The filtered list of directories.
        """

        # currently, we do not filter the directories. but you could filter the directory list like this:
        # return [dir for dir in dirs if not dir.lower().__contains__('atlas')]
        return dirs

