# Learning and Evaluating Source Code Embeddings using cuBERT

## Prerequisites

- Pytorch
- Transformers
- Tensorflow (for model conversion & tokenization)


If you did not use Docker, then you need to first activate your virtual environment and install the necessary packages.
```
pip install -r requirements.txt
```

# Create a dataset
Create a dataset by running
```
python preprocessing/preprocess.py --directory <path/to/folder/containing/all/source/code>
```
The `--directory` flag should point to the directory that contains all of the .py-files to extract methods from.
Alternatively, we make a preprocessed dataset available [here]()

# Download the pre-trained cuBERT weights
Run the following command, or download the weights from [google-research/cubert](https://github.com/google-research/google-research/tree/master/cubert).

```
gsutil -m cp \
    "gs://cubert/20200621_Python/pre_trained_model__epochs_2__length_512/model.ckpt-602440.data-00000-of-00001" \
    "gs://cubert/20200621_Python/pre_trained_model__epochs_2__length_512/model.ckpt-602440.index" \
    "gs://cubert/20200621_Python/pre_trained_model__epochs_2__length_512/model.ckpt-602440.meta" \
    /home/scc/src/model/tf_weights
```

Before fine-tuning cuBERT, we first need to convert the tensorflow weights into pytorch-friendly model weights.
Run the conversion script, specifying the correct paths.

```
python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./model/tf_weights/model.ckpt-602440.index \
  --bert_config_file ./model/config.json \
  --pytorch_dump_path ./model/pre_trained
```

# Configure paths and hyperparameters
Update the `config.py` files with the paths and hyperparameters according to your system.

# Fine-tune cuBERT

Fine-tune cuBERT by running
```
python train.py
```

# Feature extraction
Extract the fine-tuned features for the samples in the test set, saving the results to a .json file.
```
python extract_features.py
```