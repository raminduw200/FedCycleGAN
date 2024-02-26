"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
from torch.utils.data import DataLoader
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import os


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (
                dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


def prepare_dataset(opt):
    dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
    dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

    A_paths = sorted(make_dataset(dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
    B_paths = sorted(make_dataset(dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'

    num_images = len(A_paths) // opt.client_count
    # partition_len = [num_images] * opt.num_partitions
    A_paths_lists = [A_paths[i:i + num_images] for i in range(0, len(A_paths), num_images)]
    B_paths_lists = [B_paths[i:i + num_images] for i in range(0, len(B_paths), num_images)]

    return A_paths_lists, B_paths_lists


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, cid=1):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.cid = cid
        self.dataloaders = []
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.A_paths_lists, self.B_paths_lists = prepare_dataset(opt)

        for cid in range(opt.client_count):
            dataset = dataset_class(opt, self.A_paths_lists[cid], self.B_paths_lists[cid])
            print("dataset [%s] was created" % type(dataset).__name__)
            dataloader = DataLoader(
                dataset,
                batch_size=opt.batch_size,
                shuffle=not opt.serial_batches,
                # num_workers=int(opt.num_threads)
            )
            self.dataloaders.append(dataloader)

    def set_cid(self, cid):
        self.cid = cid

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        # return min(len(self.dataset), self.opt.max_dataset_size)
        return len(self.dataloaders[self.cid])

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloaders[self.cid]):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
