import requests
import torch
import numpy as np
import torch.utils.data
import torchvision


from typing import Union


def _download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        content_iter = response.iter_content(CHUNK_SIZE)
        with open(destination, "wb") as f:

            for i, chunk in enumerate(content_iter):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    print(i, end='\r')

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)

    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


class DynamicDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, wrappee):
        super().__init__()
        assert isinstance(wrappee, torch.utils.data.Dataset)
        self.wrappee = wrappee

    def __getattr__(self, name):
        return getattr(self.__dict__['wrappee'], name)

    def __len__(self):
        return len(self.wrappee)

    def __getitem__(self, idx):
        return self.wrappee[idx]


class Transformer(DynamicDatasetWrapper):
    def __init__(self, wrappee, transform=None, target_transform=None):
        super().__init__(wrappee)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.wrappee[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        return x, y


# region dataset operations


def ds_statistics(dataset: torch.utils.data.dataset.Dataset) -> dict:
    """[summary]
    
    Args:
        dataset (torch.utils.data.dataset.Dataset): dataset to analyze
    
    Returns:
        dict: a dict with the following key-value pairs:

            channel_mean: mean over the first dimension of the dataset items.

            channel_std: standard deviation over the first dimension of the 
            dataset items.

            num_classes: number of classes in the dataset. 
    """

    x, _ = dataset[0]
    if not isinstance(x, torch.Tensor):
        dataset = Transformer(dataset, torchvision.transforms.ToTensor())

    X = []
    Y = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        x = x.view(x.size(0), -1)
        X.append(x)
        Y.append(y)

    X = torch.cat(X, dim=1)

    mean = tuple(X.mean(dim=1).tolist())
    std = tuple(X.std(dim=1).tolist())

    num_classes = len(set(Y))

    return {
        'channel_mean': mean,
        'channel_std': std,
        'num_classes': num_classes
    }


def ds_random_subset(
        dataset: torch.utils.data.dataset.Dataset,
        percentage: float = None,
        absolute_size: int = None,
        replace: bool = False):
    r"""
    Represents a fixed random subset of the given dataset.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Target dataset.
        percentage (float): Percentage of target dataset to use (within [0,1]).
        absolute_size (int): Absolute size of the subset to use.
        replace (bool): Draw with replacement.

    Returns:
        A ``torch.utils.data.dataset.Dataset`` with randomly selected samples.

    .. note::
        ``percentage`` and ``absolute_size`` are mutally exclusive. So only
        one of them can be specified.
    """
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert percentage is not None or absolute_size is not None
    assert not (percentage is None and absolute_size is None)
    if percentage is not None:
        assert 0 < percentage and percentage < 1, "percentage assumed to be > 0 and < 1"
    if absolute_size is not None:
        assert absolute_size <= len(dataset)

    n_samples = int(percentage*len(dataset)
                    ) if percentage is not None else absolute_size
    indices = np.random.choice(
        list(range(len(dataset))),
        n_samples,
        replace=replace)

    indices = [int(i) for i in indices]

    return torch.utils.data.dataset.Subset(dataset, indices)


def ds_label_filter(
        dataset: torch.utils.data.dataset.Dataset,
        labels: Union[tuple, list]):
    """
    Returns a dataset with samples having selected labels.

    Args:
        dataset (torch.utils.data.dataset.Dataset): Target dataset.
        labels (tuple or list): White list of labels to use.

    Returns:
        A ``torch.utils.data.dataset.Dataset`` only containing samples having
        the selected labels.
    """
    assert isinstance(dataset, torch.utils.data.dataset.Dataset)
    assert isinstance(labels, (tuple, list)
                      ), "labels is expected to be list or tuple."
    assert len(set(labels)) == len(
        labels), "labels is expected to have unique elements."
    assert hasattr(
        dataset, 'targets'), "dataset is expected to have 'targets' attribute"
    assert set(labels) <= set(
        dataset.targets), "labels is expected to contain only valid labels of dataset"

    indices = [i for i in range(len(dataset)) if dataset.targets[i] in labels]

    return torch.utils.data.dataset.Subset(dataset, indices)


# endregion
