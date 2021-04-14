import torch

MODEL_PATH = './model/20200621_Python_pre_trained_model__epochs_2__length_512_model.ckpt-602440.index'
MODEL_CONFIG = './model/cubert_large_config.json'
MODEL_VOCAB = './model/vocab.txt'
MAX_SEQUENCE_LENGTH = 512

BATCH_SIZE = 1
EPOCHS = 5
NUM_BATCHES_TO_LOG = 100
NUM_BATCHES_UNTIL_EVAL = 500

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')