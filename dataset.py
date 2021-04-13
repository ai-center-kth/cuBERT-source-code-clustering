import torch
import config
import numpy as np
import pandas as pd

torch.multiprocessing.set_sharing_strategy('file_system')

class TripletDataset(torch.utils.data.Dataset):
    """
    For each sample (anchor) randomly chooses a positive and negative samples
    """
    def __init__(self, filename):
        self.dataset = pd.read_json(filename, lines=True)
        self.label = self.dataset.label.array
        self.labels_set = set(self.label)
        self.label_to_indices = {l: np.where(self.label == l)[0]
                                    for l in self.labels_set}

    def __getitem__(self, index):

        anchor_row = self.dataset.iloc[index]

        anchor, anchor_label = anchor_row.method, anchor_row.label

        # Randomly choose a positive sample
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(
                self.label_to_indices[anchor_label])

        positive_row = self.dataset.iloc[positive_index]
        positive = positive_row.method

        # Randomly choose a negative sample
        negative_label = np.random.choice(
            list(self.labels_set - set([anchor_label])))
        
        negative_index = np.random.choice(
            self.label_to_indices[negative_label])
        
        negative_row = self.dataset.iloc[negative_index]
        negative = negative_row.method

        return (anchor, positive, negative)

    def __len__(self):
        return len(self.dataset)