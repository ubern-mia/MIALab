"""The loading module holds classes to load data."""
from abc import ABCMeta, abstractmethod
import os


class FilePathGenerator(metaclass=ABCMeta):
    """TODO"""

    @staticmethod
    @abstractmethod
    def get_full_file_path(id_: str, root_dir: str, file_key, file_extension: str) -> str:
        """

        Args:
            id_ ():
            root_dir ():
            file_key ():
            file_extension ():

        Returns:
            str:
        """
        raise NotImplementedError()


class FileSystemDataCrawler:
    """Represents a file system data crawler.

    TODO
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

    >>> crawler = FileSystemDataCrawler('root_dir', dir_filter='Patient', file_extension='.mha')
    >>> for id_, path in crawler.data_dir.items():
    >>>     print(id_, path)
    Patient1 ./Patient1
    Patient2 ./Patient2
    """
    
    def __init__(self,
                 root_dir: str,
                 file_keys: list,
                 file_path_generator: FilePathGenerator,
                 dir_filter: str = None,
                 file_extension: str='.nii.gz'):
        """Initializes a new instance of the SITKImageLoader class.

        Args:
            root_dir (str): The path to the root directory, which contains subdirectories with the data.
            file_keys (list): A list of strings, which represent file suffixes. TODO
            file_path_generator (FilePathGenerator): TODO
            dir_filter (str): A string to filter the data directories (contains filter).
            file_extension (str): The data file extension (with or without dot).
        """
        super().__init__()

        self.root_dir = root_dir
        self.dir_filter = dir_filter
        self.file_keys = file_keys
        self.file_path_generator = file_path_generator
        self.file_extension = file_extension if file_extension.startswith('.') else '.' + file_extension

        # dict with key=id (i.e, directory name), value=path to data directory
        self.data = {}  # dict with key=id (i.e, directory name), value=dict with key=file_keys and value=path to file

        data_dir = self._crawl_directories()
        self._crawl_data(data_dir)

    def _crawl_data(self, data_dir: dict):
        """Crawls the data inside a directory."""

        for id_, path in data_dir.items():
            data_dict = {id_: path}  # init dict with id_ pointing to path
            for item in self.file_keys:
                file_path = self.file_path_generator.get_full_file_path(id_, path, item, self.file_extension)
                data_dict[item] = file_path

            self.data[id_] = data_dict

    def _crawl_directories(self):
        """Crawls the directories, which contain data."""

        if not os.path.isdir(self.root_dir):
            raise ValueError('root_dir should point to an existing directory')

        # search the root directory for data directories
        data_dirs = next(os.walk(self.root_dir))[1]

        if self.dir_filter:
            # filter the data directories
            data_dirs = [data_dir for data_dir in data_dirs if data_dir.__contains__(self.dir_filter)]

        return {
            data_dir: os.path.join(self.root_dir, data_dir)
            for data_dir in data_dirs
            if any(file.endswith(self.file_extension) for file  # check if directory contains data files
                   in os.listdir(os.path.join(self.root_dir, data_dir)))
        }
