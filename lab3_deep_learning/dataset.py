import numpy as np
from mnist import MNIST # pip install python-mnist
import logging
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

def build_dataset_and_loader(batch_size: int, partition: str, logger: logging.Logger, data_dir="./data/"):
    assert partition in ('training', 'testing'), f"{partition} is an invalid partition"
    dataset = MNISTDataset(partition, data_dir, logger)
    dataloader = DataLoader(dataset, batch_size, shuffle=partition == 'training', num_workers=2,
                            collate_fn=dataset.collate_fnc)
    return dataset, dataloader

class MNISTDataset(Dataset):
    imsize = 28

    def __init__(self, partition: str, mnist_dir: str, logger: logging.Logger):
        assert partition in ('training', 'testing'), f"{partition} is an invalid partition"
        mnist = MNIST(mnist_dir)
        raw_dataset_parser = getattr(mnist, f"load_{partition}")
        self.images, self.labels = raw_dataset_parser()
        self.nimages = len(self.images)
        logger.info(f"Loaded {self.nimages} images for {partition}")

    def __len__(self):
        return self.nimages

    def __getitem__(self, index: int) -> Dict:
        """
        TODO: Return a single data point (a pair of image & label) at the position `index` in the dataset

        :param index: of the data point to be accessed
        :return: {
            'image': np.ndarray, shape (28 * 28), NOTE: images loaded from raw_dataset is flatten to a 1-d array
            'label': int
        }
        """
        #TODO: implement this method
        
        img = self.images[index]
        label = self.labels[index]
        
        return {'image': img, 'label': label}

    @staticmethod
    def collate_fnc(data_batch: List[Dict]) -> Dict:
        """
        TODO: Recipe for batching individual data_dict into a mini batch

        :param data_batch: a List of N dict, each dict is {
            'image': np.ndarray, shape (28 * 28),
            'label': int
        }
        :return: a single dict {
            'image': np.ndarray, shape (N, 28 * 28),
            'label': np.ndarray, shape (N)
        }
        """
        images = []
        labels = []
        #TODO: implement this method
        for i in range(0, len(data_batch)):
            images.append(data_batch[i]['image'])
            labels.append(data_batch[i]['label'])

        return {'image': images, 'label': labels}