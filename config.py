import torch

MODEL_PATH = './model/cubert_large.bin'
MODEL_CONFIG = './model/cubert_large_config.json'
MODEL_VOCAB = './model/vocab.txt'
MAX_SEQUENCE_LENGTH = 512

BATCH_SIZE = 1
EPOCHS = 5
NUM_BATCHES_TO_LOG = 100
NUM_BATCHES_UNTIL_EVAL = 1000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')