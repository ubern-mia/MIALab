"""The loading module holds classes to load data."""
import os


class DataLoaderBase:
    """Represents a base class for data loading."""
    
    def __init__(self):
        """Initializes a new instance of the DataLoaderBase class."""
    
    def __iter__(self):
        """Gets an iterator object.
        
        Returns:
            DataLoaderBase: Itself.
        """
        return self
    
    def __next__(self):
        """Gets the next data item.
        
        Returns:
            object: The data item.
        """
        raise NotImplementedError()


class FileSystemDataLoader(DataLoaderBase):
    """Represents a file system data loader.

    Given a dictionary, where keys represent an identifier and values the paths to directories with the data,
    the data can be loaded sequentially.
    """
    
    def __init__(self, 
                 data: dict,
                 file_extension: str='.nii.gz'):
        """Initializes a new instance of the FileSystemDataLoader class.

        Args:
            data (dict): The data dictionary. Keys (str) represent an identifier and 
                values (str) the paths to directories with the data.
            file_extension (str): The data's file extension (with or without the dot).
        """
        self.data = data
        
    def __next__(self):
        """Gets the next data item.
        
        Returns:
            object: The data item.
        """
        raise NotImplementedError()


class FileSystemCrawler:
    """Represents file system crawler.

    Given a root directory, the crawler searches all subdirectories which contain images.
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
