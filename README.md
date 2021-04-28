#

## Prerequisites

- Pytorch
- Transformers
- Tensorflow (for model conversion & tokenization)

Activate your virtual environment and install the necessary packages.
```
pip install -r requirements.txt
```

# Creating a dataset
Create a dataset by running
```
python preprocessing/preprocess.py --directory <path/to/folder/containing/all/source/code>
```
The `--directory` flag should point to the directory which contains all of the .py-files to extract methods from.

# Converting Pretrained cuBERT
Before fine-tuning cuBERT, we first need to download and convert a pretrained cuBERT model to use.
Download a pretrained model from [google-research/cubert](https://github.com/google-research/google-research/tree/master/cubert), then run the conversion script to obtain pytorch-friendly model weights.

```
python convert_tf_checkpoint_to_pytorch \
  --tf_checkpoint_path <path/to/model/bert_model.index> \
  --bert_config_file <path/to/config/bert_config.json> \
  --pytorch_dump_path <path/to/output/cuBERT.bin>
```

# Configure paths and hyperparameters
Update the `config.py` files with the paths and hyperparameters according to your system.

# Fine-tune cuBERT

Fine-tune cuBERT by running
```
python train.py
```

# Feature extraction
```
python extract_features.py
```