import os
import cv2 as cv
import numpy as np

from megengine.data import DataLoader, RandomSampler, SequentialSampler
from megengine.data.dataset import Dataset
import megengine.data.transform as T


# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = T.Compose([
    # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    T.RandomHorizontalFlip(),  # randomly flip image horizontally
    T.ToMode('CHW')
    ])

# loader for evaluation, no horizontal flip
eval_transformer = T.Compose([
    # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    T.ToMode('CHW')
    ])


class SIGNSDataset(Dataset):
    """
    A standard MegEngine definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir):
        """
        Store the filenames of the jpgs to use.
        Args:
            data_dir: (string) directory containing the dataset
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]

        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (np.array) image
            label: (int) corresponding label of image
        """
        image = cv.imread(self.filenames[idx], cv.IMREAD_COLOR)  # image as np.array
        image = np.array(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                ds = SIGNSDataset(path)
                dl = DataLoader(ds, 
                                sampler=RandomSampler(ds, batch_size=params.batch_size),
                                transform=train_transformer,
                                num_workers=params.num_workers)
            else:
                ds = SIGNSDataset(path)
                dl = DataLoader(ds, 
                                sampler=SequentialSampler(ds, batch_size=params.batch_size),
                                transform=eval_transformer,
                                num_workers=params.num_workers)

            dataloaders[split] = dl

    return dataloaders