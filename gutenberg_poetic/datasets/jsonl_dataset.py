import jsonlines

from typing import Text

from torch.utils.data import Dataset


class JsonlDataset(Dataset):

    def __init__(self,
                 path: Text,
                 preprocess=None):
        super().__init__()

        self.preprocess = preprocess
        with jsonlines.open(path) as f:
            self.data = list(f)
        self.total = len(self.data)

    def __getitem__(self, idx: int):
        if self.preprocess:
            return self.preprocess(self.data[idx])
        else:
            return self.data[idx]

    def __len__(self):
        return self.total

