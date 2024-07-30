import torch
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

### We load our MNIST dataset using the torchvision package.
### We need to specify the batch size for the dataloader first¶

### We load our MNIST dataset using the torchvision package.
### We need to specify the batch size for the dataloader first¶
# batch_size=64

class MNISTLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Load MNIST Training data
        # Note that we shuffle the training data
        self.train_loader = self._load_train_data()
        # Load MNIST Test data
        # Note that we DO NOT shuffle the test data
        self.test_loader = self._load_test_data()

    def _load_train_data(self):
        return DataLoader(
            MNIST(
                root='./data',
                train=True,
                download=True,
                transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True
        )

    def _load_test_data(self):
        return DataLoader(
            MNIST(
                root='./data',
                train=False,
                download=True,
                transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False
        )


###### MNIST FASHION DATASET ##########

class FashionMNISTLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Load MNIST Training data
        # Note that we shuffle the training data
        self.train_loader = self._load_train_data()
        # Load MNIST Test data
        # Note that we DO NOT shuffle the test data
        self.test_loader = self._load_test_data()

    def _load_train_data(self):
        return DataLoader(
            FashionMNIST(
                root='./data',
                train=True,
                download=True,
                transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True
        )
        

    def _load_test_data(self):
        return DataLoader(
            FashionMNIST(
                root='./data',
                train=False,
                download=True,
                transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False
        )



import os
import torch
from PIL import Image
from scipy.io import loadmat
import torchvision.transforms as transforms

class FreyFaceLoader(torch.utils.data.Dataset):
    # data_file: available at https://cs.nyu.edu/~roweis/data/frey_rawface.mat
    data_file = 'frey_rawface.mat'

    def __init__(self, root, batch_size):
        super(FreyFaceLoader, self).__init__()
        self.root = root
        self.batch_size = batch_size
        self.transform = transforms.ToTensor()
        if not self._check_exists():
            raise RuntimeError('Dataset do not found in the directory \"{}\". \nYou can download FreyFace '
                               'dataset from https://cs.nyu.edu/~roweis/data/frey_rawface.mat '.format(self.root))
        self.data = loadmat(os.path.join(self.root, self.data_file))['ff'].T

        # Load MNIST Train data
        self.train_loader = self._load_train_data()

        # Load MNIST Test data
        # Note that we DO NOT shuffle the test data
        self.test_loader = self._load_test_data()

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 20)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.data_file))

    def _load_train_data(self):
        indices = torch.randperm(len(self)).tolist()
        train_indices = indices[400:]
        train_dataset = torch.utils.data.Subset(dataset=self, indices=train_indices)
        return torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

    def _load_test_data(self):
        indices = torch.randperm(len(self)).tolist()
        test_indices = indices[:400]
        test_dataset = torch.utils.data.Subset(dataset=self, indices=test_indices)
        return torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)


# Build the data input pipeline
# entire_dataset = FreyFaceDataset(root='./data/FreyFace', batch_size=128)


def data_processing(data):
    # If data is a list of features,labels
    if isinstance(data, list):
        features=data[0]
    else:
        features=data
    
    # We only want features
    return features
    
    ### Set our seed and other configurations for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# entire_dataset = FreyFaceDataset(root='./data/frey_rawface.mat', batch_size=64)
