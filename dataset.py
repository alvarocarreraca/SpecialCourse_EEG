import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, xList, labels):
        self.labels = labels
        self.xList = xList

    def __len__(self):
        return len(self.xList)

    def __getitem__(self, index):
        X = self.xList[index]
        y = self.labels[index]

        return X, y
