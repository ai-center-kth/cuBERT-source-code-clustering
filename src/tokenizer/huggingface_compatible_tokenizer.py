#
# Thanks to DNGRos for this huggingface/transformers compatible version
# https://github.com/google-research/google-research/issues/582
#
import os
import torch
import config
import collections
from typing import *
from transformers import BertTokenizer
from tokenizer.cubert_tokenizer import CuBertTokenizer
from tokenizer.python_tokenizer import PythonTokenizer
from tensor2tensor.data_generators import text_encoder


def combine_tokenizer_with_subword(
    initial_tokenizer: CuBertTokenizer,
    subword_tokenizer: text_encoder.SubwordTextEncoder,
) -> Callable[[str], List[str]]:
    # Try to match the functionality at 
    # https://github.com/google-research/google-research/blob/50c6cd94b5/cubert/code_to_subtokenized_sentences.py#L111-L118
    
    def tokenize(string: str) -> List[str]:
        toks = initial_tokenizer.tokenize(string)
        tokens = flatten_list(
            subword_tokenizer.decode_list(
                subword_tokenizer.encode_without_tokenizing(token)
            )
            for token in toks
        )
        return tokens
    return tokenize


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class CuBertHugTokenizer(BertTokenizer):
    # A hacky version that seems to work at least for python
    def __init__(
        self,
        vocab_file: str,
    ):
        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=False,
            do_basic_tokenize=True,
            unk_token="[UNK]_",
            sep_token="[SEP]_",
            pad_token="<pad>_",
            cls_token="[CLS]_",
            mask_token="[MASK]_",
        )

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    vocab_file)
            )
        self.vocab = self.load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.first_tokenizer = PythonTokenizer()
        self.subword_tokenizer = text_encoder.SubwordTextEncoder(str(vocab_file))
        self._combined_func = combine_tokenizer_with_subword(
            self.first_tokenizer, self.subword_tokenizer)

    def __call__(self, text):
        return super().__call__(
            text,
            padding='max_length',
            truncation='longest_first',
            max_length=config.MAX_SEQUENCE_LENGTH
        )

    def mask_tokens(self, inputs: dict, mlm_probability=0.1):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 10% MASK.
        """
        for k,v in inputs.items():
            inputs[k] = torch.tensor(v)
        
        labels = inputs['input_ids']
        labels = labels.clone().detach()
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        assert 1-mlm_probability >= 0
        # 10% of the time (given by mlm_probability), we replace input tokens with tokenizer.mask_token ([MASK_])
        probability_matrix = torch.full(labels.shape, 1-mlm_probability)
        
        special_tokens_mask = self.get_special_tokens_mask(labels.tolist(), already_has_special_tokens=True) # Avoid masking special tokens
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        indices_replaced = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool() & masked_indices
        
        # Set token ID to -100 for the tokens to ignore when computing MLM loss
        # See https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_bert.html#BertForMaskedLM
        labels[~indices_replaced] = -100   
        inputs['input_ids'][indices_replaced] = self.convert_tokens_to_ids(self.mask_token)
        inputs['labels'] = labels
        return inputs
    
    @property
    def do_lower_case(self):
        return False

    def _tokenize(self, text):
        return self._combined_func(text)

    def convert_tokens_to_string(self, tokens):
        raise NotImplementedError

    def _convert_token_to_id(self, token):
        return self.subword_tokenizer._subtoken_string_to_id[token]

    def load_vocab(self, vocab_file):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(vocab_file, "r", encoding="utf-8") as reader:
            tokens = reader.readlines()
        for index, token in enumerate(tokens):
            token = token.rstrip("\n")
            vocab[token] = index
        return vocab