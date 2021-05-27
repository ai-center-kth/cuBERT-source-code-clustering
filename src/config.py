import torch


MODEL_PATH = "./model/pre_trained/pytorch_model.bin"
MODEL_CONFIG = "./model/config.json"
MODEL_CHECKPOINT_PATH = "./model/checkpoints/cubert.ckpt"
MODEL_VOCAB = "./model/vocab.txt"
DATASET_DIR = "./data/small"
LOG_DIR = "./logs/small"
RESULT_DIR = "./results"
MAX_SEQUENCE_LENGTH = 256

LEARNING_RATE = 3e-5
BATCH_SIZE = 4
EPOCHS = 1
NUM_BATCHES_UNTIL_LOG = 100
NUM_BATCHES_UNTIL_EVAL = 3000

# Parameters used by the deep robust clustering and unsupervised frameworks
NUM_CLUSTERS = 5

# Parameters used only by the deep robust clustering framework
LAMBDA = 0.5            # Regularization parameter
TEMPERATURE_AF = 0.5    # Temperature parameter for assignment features
TEMPERATURE_AP = 1.0    # Temperature parameter for assignemtn probabilies


# Device to use for training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")