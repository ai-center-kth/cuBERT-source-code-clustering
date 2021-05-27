import torch
import pickle
import numpy as np
import pandas as pd

from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
torch.multiprocessing.set_sharing_strategy('file_system')

class TripletDataset(torch.utils.data.Dataset):
    """
    Class for generating method triplets (anchor, positive, negative) from the dataset.
    Used in the Triplet framework.
    """

    def __init__(self, filename, tokenizer):
        self.dataset = pd.read_json(filename, lines=True)
        self.tokenizer = tokenizer
        self.index_set = set(self.dataset.index.array)
        self.method_vocab = Vocabulary(filename.replace('json', 'pickle'))
        
    def __getitem__(self, index):

        anchor_sample, positive_sample, negative_sample = {}, {}, {}
        
        anchor_row = self.dataset.iloc[index]
        anchor_sample = self.tokenizer(anchor_row.method)
        anchor_sample["method_name"] = anchor_row.method_name
        

        # Get subwords in method name to find similar methods
        methodname_subwords = [
            x for x in anchor_row.method_name.split('_') if x]

        # Get indices of other methods that contain at least one of the subwords
        positive_indices = []
        for subword in methodname_subwords:
            indices = self.method_vocab.lookup_indices(subword)
            if indices is not None:
                positive_indices.extend(indices)

        positive_indices = list(set(positive_indices) - set([index]))
        
        if len(positive_indices) > 1:
            # Get names of methods which have at least one matching subword
            positive_matches = pd.DataFrame(
                self.dataset.iloc[positive_indices].method_name)

            # Compute bleu score
            positive_matches["score"] = positive_matches.method_name.apply(
                lambda x: sentence_bleu([x], anchor_row.method_name, smoothing_function=SmoothingFunction().method5))

            if positive_matches.score.max() > 0.5:
                positive_matches = positive_matches[positive_matches.score > 0.5]

            positive_matches["norm_score"] = positive_matches.score / \
                positive_matches.score.sum()

            positive_index = np.random.choice(
                positive_matches.index, p=positive_matches.norm_score)

            positive_row = self.dataset.iloc[positive_index]
            positive_sample = self.tokenizer(positive_row.method)
            positive_sample["method_name"] = positive_row.method_name
            
        else:
            # On the very rare occasion that we cant find any other method with a similar name
            positive_sample = self.tokenizer(anchor_row.method)
            positive_sample["method_name"] = anchor_row.method_name
                               
            
        # Randomly choose one negative sample that does not have similar label
        negative_indices = list(self.index_set - set(positive_indices))
        negative_index = np.random.choice(negative_indices)
        negative_row = self.dataset.iloc[negative_index]
        negative_sample = self.tokenizer(negative_row.method)
        negative_sample["method_name"] = negative_row.method_name
        
        
        return { "anchor": anchor_sample, "positive": positive_sample, "negative": negative_sample }

    def __len__(self):
        return len(self.dataset)


class DRCDataset(torch.utils.data.Dataset):
    """
    Used in the Deep Robust Clustering framework.
    Class for generating method tuples (anchor, positive) from the dataset.
    """

    def __init__(self, filename, tokenizer):
        self.dataset = pd.read_json(filename, lines=True)
        self.tokenizer = tokenizer
        self.index_set = set(self.dataset.index.array)
        self.method_vocab = Vocabulary(filename.replace('json', 'pickle'))
        
    def __getitem__(self, index):

        anchor_sample, positive_sample, negative_sample = {}, {}, {}
        
        anchor_row = self.dataset.iloc[index]
        anchor_sample = self.tokenizer(anchor_row.method)
        anchor_sample["method_name"] = anchor_row.method_name
        

        # Get subwords in method name to find similar methods
        methodname_subwords = [
            x for x in anchor_row.method_name.split('_') if x]

        # Get indices of other methods that contain at least one of the subwords
        positive_indices = []
        for subword in methodname_subwords:
            indices = self.method_vocab.lookup_indices(subword)
            if indices is not None:
                positive_indices.extend(indices)

        positive_indices = list(set(positive_indices) - set([index]))
        
        if len(positive_indices) > 1:
            # Get names of methods which have at least one matching subword
            positive_matches = pd.DataFrame(
                self.dataset.iloc[positive_indices].method_name)

            # Compute bleu score
            positive_matches["score"] = positive_matches.method_name.apply(
                lambda x: sentence_bleu([x], anchor_row.method_name, smoothing_function=SmoothingFunction().method5))

            if positive_matches.score.max() > 0.5:
                positive_matches = positive_matches[positive_matches.score > 0.5]

            positive_matches["norm_score"] = positive_matches.score / \
                positive_matches.score.sum()

            positive_index = np.random.choice(
                positive_matches.index, p=positive_matches.norm_score)

            positive_row = self.dataset.iloc[positive_index]
            positive_sample = self.tokenizer(positive_row.method)
            positive_sample["method_name"] = positive_row.method_name
            
        else:
            # On the very rare occasion that we cant find any other method with a similar name
            positive_sample = self.tokenizer(anchor_row.method)
            positive_sample["method_name"] = anchor_row.method_name
        
        return { "anchor": anchor_sample, "positive": positive_sample }

    def __len__(self):
        return len(self.dataset)


class UnsupervisedDataset(torch.utils.data.Dataset):
    """
    Class for generating method samples from the dataset.
    Used with the Unsupervised Framework.
    """

    def __init__(self, filename, tokenizer, mlm=True, mlm_probability=0.1):
        self.dataset = pd.read_json(filename, lines=True)
        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability
    
    def __getitem__(self, index):

        row = self.dataset.iloc[index]
        
        tokenized_method = self.tokenizer(row.method)
        
        if self.mlm:
            inputs = self.tokenizer.mask_tokens(tokenized_method, mlm_probability=self.mlm_probability)
        
        inputs['method_name'] = row.method_name
        
        return inputs

    def __len__(self):
        return len(self.dataset)


class Vocabulary(object):
    """
    Vocabulary for looking up word-indices mappings of the method names in the dataset.
    The mapping makes it more efficient of finding similar methods by their name.
    """    
    def __init__(self, word_path: str):
        self.word2idx = self.load_dictionary(word_path)
        
    def lookup_indices(self, word):
        if word not in self.word2idx:
            return None
        else:
            return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def load_dictionary(self, fname):
        with open(fname, 'rb') as file:
            dictionary = pickle.load(file)
        return dictionary

def data_collator(samples):
    """Collator function for stacking and converting the samples to tensors.

    Args:
        samples (List[dict]): A list of dictionaries where the keys are the sample identities, i.e. { anchor: ..., positive: ..., negative:... }

    Returns:
        [torch.Tensor]: A tensorized batch
    """    

    batch = {key: defaultdict(list) for key in samples[0].keys()}
    for sample in samples:
        # Tensorize
        for key in sample.keys():
            batch[key]['input_ids'].append(torch.tensor(sample[key]['input_ids'], dtype=torch.int64))
            batch[key]['attention_mask'].append(torch.tensor(sample[key]['attention_mask'], dtype=torch.int64))
            batch[key]['token_type_ids'].append(torch.tensor(sample[key]['token_type_ids'], dtype=torch.int64))
            batch[key]['method_name'].append(sample[key]['method_name'])

    # Convert lists into tensors by stacking
    for key in batch.keys():
        batch[key]['input_ids'] = torch.stack(batch[key]['input_ids'])
        batch[key]['attention_mask'] = torch.stack(batch[key]['attention_mask'])
        batch[key]['token_type_ids'] = torch.stack(batch[key]['token_type_ids'])

    return batch