#

## Prerequisites

- Pytorch
- Transformers
- Tensorflow (for model conversion & tokenization)

```
pip install -r requirements.txt
```

# Converting Pretrained cuBERT

First we need to convert a pretrained cuBERT model to use it with pytorch+transformers.
Download a pretrained model from [google-research/cubert](https://github.com/google-research/google-research/tree/master/cubert)

```
transformers-cli convert --model_type bert \
  --tf_checkpoint path/to/model/bert_model.ckpt \
  --config path/to/config/bert_config.json \
  --pytorch_dump_output path/to/output/cuBERT.bin
```

# Update config.py
Update the `config.py` files with the paths and hyperparameters according to your system.

# Fine-tune cuBERT

Fine-tune cuBERT by running
```
python train.py
```