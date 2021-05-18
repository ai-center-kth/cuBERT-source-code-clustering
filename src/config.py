import torch

MODEL_PATH = "./model/model/pre_trained/pytorch_model.bin"
MODEL_CONFIG = "./model/pre_trained/config.json"
MODEL_CHECKPOINT_PATH = "./model/checkpoints/cubert.ckpt"
MODEL_VOCAB = "./model/vocab.txt"
DATASET_DIR = "./data/small"
LOG_DIR = "./logdir"
RESULT_DIR = "./results"
MAX_SEQUENCE_LENGTH = 256

LEARNING_RATE = 3e-5
BATCH_SIZE = 4
EPOCHS = 1
NUM_BATCHES_UNTIL_LOG = 100
NUM_BATCHES_UNTIL_EVAL = 1000

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
