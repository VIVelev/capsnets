import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__all__ = [
    'BATCH_SIZE',
    'TEST_BATCH_SIZE',
    'NUM_WORKERS',

    'get_transform',

    'get_train_set'
    'get_test_set',
    'get_sets',

    'imshow',

    'get_train_set_loader',
    'get_test_set_loader',
    'get_set_loaders',
]


# ====================================================================================================
# ====================================================================================================

BATCH_SIZE = 128
TEST_BATCH_SIZE = 256
NUM_WORKERS = 4

# ====================================================================================================

def get_transform():
    pil_to_tensor_transform = transforms.ToTensor()

    return transforms.Compose([
        pil_to_tensor_transform,
    ])

# ====================================================================================================

def get_train_set(path='../datasets', download=True):
    return datasets.CIFAR10(path, train=True, download=download, transform=get_transform())

def get_test_set(path='../datasets', download=True):
    return datasets.CIFAR10(path, train=False, download=download, transform=get_transform())

def get_sets(path='../datasets', download=True):
    return (
        get_train_set(path, download),
        get_test_set(path, download),
    )

# ====================================================================================================

def imshow(tensor):
    plt.imshow(np.transpose(tensor, (1, 2, 0)))
    plt.show()

# ====================================================================================================

def get_train_set_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    return data.DataLoader(get_train_set(), batch_size=batch_size, shuffle=True, num_workers=num_workers)

def get_test_set_loader(batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS):
    return data.DataLoader(get_test_set(), batch_size=batch_size, shuffle=False, num_workers=num_workers)

def get_set_loaders(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, num_workers=NUM_WORKERS):
    return (
        get_train_set_loader(batch_size, num_workers),
        get_test_set_loader(test_batch_size, num_workers),
    )

# ====================================================================================================
