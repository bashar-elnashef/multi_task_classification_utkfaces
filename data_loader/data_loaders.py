from torchvision import datasets, transforms
from base import BaseDataLoader
from utils import util
from dataset.dataset import UTKFace


class UTKFaceDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True, url=None):
        trsfm = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ])
        trsfm = transforms.Compose([transforms.RandomRotation(30),
                                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                                    transforms.ColorJitter(brightness=0.5),
                                    transforms.RandomVerticalFlip(),
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self.data_dir = data_dir
        self.dataset = UTKFace(self.data_dir, download=True, transform=trsfm, url=url)
        print()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
