import torch

MODEL_PATH = './model/pytorch_model/cubert.bin'
MODEL_CONFIG = './model/pytorch_model/config.json'
MODEL_CHECKPOINT_PATH = './model/checkpoints/cubert.ckpt'
MODEL_VOCAB = './model/vocab.txt'
MAX_SEQUENCE_LENGTH = 256

LEARNING_RATE = 3e-5
BATCH_SIZE = 4
EPOCHS = 1
NUM_BATCHES_TO_LOG = 100
NUM_BATCHES_UNTIL_EVAL = 1000

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')