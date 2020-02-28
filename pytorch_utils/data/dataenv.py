import numpy as np
import pickle


from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data.dataset import Subset

class DatasetFactory():
    def __init__(self,
                 ds_name_to_fn_and_kwargs
                 ):

        self.ds_name_to_fn_and_kwargs = ds_name_to_fn_and_kwargs

    def __call__(self,
                 ds_name: str):

        ds_name = str(ds_name)
        assert ds_name in self.ds_name_to_fn_and_kwargs

        try:
            ds_fn, kwargs = self.ds_name_to_fn_and_kwargs[ds_name]

        except KeyError:
            raise KeyError('Expected ds_name to be in {} but got {}'.format(
                list(self.ds_name_to_fn_and_kwargs.keys()),
                ds_name)
            )

        return ds_fn(**kwargs)

    def __iter__(self):
        return iter(self.ds_name_to_fn_and_kwargs)


class StratifiedShuffleSplitDatasetFactory:
    def __init__(
        self, 
        ds_factory, 
        split_cfg, 
        split_indices_path, 
        num_splits=20):

        self.ds_factory = ds_factory 
        self.split_cfg = split_cfg
        self.pth = Path(split_indices_path)
        self.num_splits = num_splits

        self.indices = self.load_indices() 

    def load_indices(self):
        if self.pth.is_file():
            with open(str(self.pth), 'br') as fid:
                indices = pickle.load(fid)

        else:
            indices = {}

        for ds_name, needed_num_samples in self.split_cfg.items():
            for num_samples in needed_num_samples:
                if (ds_name, num_samples) in indices:
                    # If we get here then StratifiedShuffleSplitFactory  
                    # was called on an index file (self.pth) which has been 
                    # created by an previous call with a different value 
                    # for num_samples. We do not allow this. 
                    assert len(indices[(ds_name, num_samples)]) == self.num_splits
                else:
                    ds = self.ds_factory(ds_name)
                    Y = [ds[i][1] for i in range(len(ds))]

                    s = StratifiedShuffleSplit(
                        self.num_splits,
                        train_size=num_samples,
                        test_size=None)

                    I = [i.tolist() for i, _ in s.split(Y, Y)]

                    indices[(ds_name, num_samples)] = I

        with open(str(self.pth), 'bw') as fid:
            pickle.dump(indices, fid)

        return indices

    def __call__(self, ds_name, num_samples):
        ds = self.ds_factory(ds_name)

        with open(str(self.pth), 'br') as fid:
            cache = pickle.load(fid)

        I = cache[ds_name, num_samples]

        return [Subset(ds, indices=i) for i in I]
