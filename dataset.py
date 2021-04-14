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
        anchor_sample = {"method": anchor_row.method, "method_name": anchor_row.method_name, "label": anchor_row.label}

        # Randomly choose a positive sample
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(
                self.label_to_indices[anchor_sample['label']])

        positive_row = self.dataset.iloc[positive_index]
        positive_sample = {"method": positive_row.method, "method_name": positive_row.method_name, "label": positive_row.label}

        # Randomly choose a negative sample
        negative_label = np.random.choice(
            list(self.labels_set - set([anchor_sample['label']])))
        
        negative_index = np.random.choice(
            self.label_to_indices[negative_label])
        
        negative_row = self.dataset.iloc[negative_index]
        negative_sample = {"method": negative_row.method, "method_name": negative_row.method_name, "label": negative_row.label}

        return (anchor_sample, positive_sample, negative_sample)

    def __len__(self):
        return len(self.dataset)