"""todo(fabianbalsiger): comment"""
import os


class IDataLoader:

    def __int__(self):
        pass


class FileSystemDataLoader(IDataLoader):
    """Represents an image loader.

    Given a root directory, the loader searches all subdirectories which contain images.
    """

    def __init__(self,
                 root_dir: str,
                 img_dir_filter: str=None,
                 img_file_extension: str='.nii.gz'):
        """Initializes a new instance of the FileSystemDataLoader class.

        Args:
            root_dir (str): The path to the root directory, which contains subdirectories with the images.
            img_dir_filter (str): A string, which filter the image directories.
            img_file_extension (str): The image file extension (with or without the dot).

        Examples:
            Suppose we have the following directory structure:
            root_dir/Patient1
                /Image.mha
                /GroundTruth.mha
                /Image.mha
                /GroundTruth.mha
            root_dir/Patient2
                /Image.mha
                /GroundTruth.mha
            root_dir/Information
                /Atlas.mha

        >>> loader = FileSystemDataLoader(".", 'Patient', '.mha')
        >>> for k, v in loader.image_path_dict.items():
        >>>     print(k, v)
        Patient1 ./Patient1
        Patient2 ./Patient2
        """
        self.root_dir = root_dir
        self.img_dir_filter = img_dir_filter
        self.img_file_extension = img_file_extension

        if not os.path.isdir(self.root_dir):
            raise ValueError("root_dir should point to an existing directory")

        # search the root directory for image directories
        image_dirs = next(os.walk(self.root_dir))[1]

        if self.img_dir_filter:
            # filter the image directories
            image_dirs = [img_dir for img_dir in image_dirs if img_dir.__contains__(self.img_dir_filter)]

        self.image_path_dict = {
            img_dir: os.path.join(self.root_dir, img_dir)
            for img_dir in image_dirs
            if any(file.endswith(self.img_file_extension) for file  # check if directory contains image files
                   in os.listdir(os.path.join(self.root_dir, img_dir)))
        }
