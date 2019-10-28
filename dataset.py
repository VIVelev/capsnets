import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

__all__ = [
    'BATCH_SIZE',
    'TEST_BATCH_SIZE',
    'get_transform',
    'get_sets',
    'imshow',
    'get_set_loaders',
]


BATCH_SIZE = 128
TEST_BATCH_SIZE = 256

def get_transform():
    pil_to_tensor_transform = transforms.ToTensor()

    return transforms.Compose([
        pil_to_tensor_transform,
    ])

def get_sets(path='./datasets', download=True):
    # Data transform
    transform = get_transform()

    # Train set
    train_set = datasets.MNIST(path, train=True, download=download, transform=transform)
    # Test set
    test_set = datasets.MNIST(path, train=False, download=download, transform=transform)

    return train_set, test_set

def imshow(tensor):
    plt.imshow(tensor.squeeze(), cmap='binary')
    plt.show()

def get_set_loaders(batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, num_workers=4):
    # Get sets
    train_set, test_set = get_sets()

    # Train set loader
    train_set_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # Test set loader
    test_set_loader = data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)

    return train_set_loader, test_set_loader
