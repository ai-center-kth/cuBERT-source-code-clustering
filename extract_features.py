import json
import torch
import config
import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import paired_cosine_distances

from tqdm import tqdm
from matplotlib import ticker
from dataset import TripletDataset
from transformers import BertForPreTraining, BertConfig
from tokenizer.huggingface_compatible_tokenizer import CuBertHugTokenizer

logging.getLogger().setLevel(logging.INFO)

def extract():
    
    # Load fine tuned model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./model/checkpoints/scubert.ckpt', config=model_config).to('cuda')
    model.eval()
    # print(model)

    logging.info(f'Loaded model on {config.DEVICE}')

    # Set up tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    test_ds = TripletDataset('./data/test.json')
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size = config.BATCH_SIZE, num_workers = 2)

    # Set loss function
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2).to(config.DEVICE)


    total_loss, total_positive_similarity, total_negative_similarity, data_cnt = 0, 0, 0, 0
    with open('./results/test.json', 'w') as writer:
        for batch_idx, (anchor, positive, negative) in tqdm(enumerate(test_dataloader, 1), desc=f"Evaluating", total=test_dataloader.__len__()):

            batch_count = np.shape(anchor['method'])[0]

            # Get anchor features
            inputs = tokenizer(anchor['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            del input_ids
            del token_type_ids
            del attention_mask
            anchor_rep = outputs.hidden_states[-1][:,0,:] # (batch_size, sequence_length, hidden_size)
            del outputs
            # Get positive features
            inputs = tokenizer(positive['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            del input_ids
            del token_type_ids
            del attention_mask
            positive_rep = outputs.hidden_states[-1][:,0,:] # (batch_size, sequence_length, hidden_size)
            del outputs
            # Get negative features
            inputs = tokenizer(negative['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            del input_ids
            del token_type_ids
            del attention_mask
            negative_rep = outputs.hidden_states[-1][:,0,:] # (batch_size, sequence_length, hidden_size)
            del outputs
            # Compute loss
            loss = loss_fn(anchor_rep, positive_rep, negative_rep)
            
            # Compute similarities within batch
            positive_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                        positive_rep.detach().to('cpu').numpy()))
            negative_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                        negative_rep.detach().to('cpu').numpy()))
            data_cnt += batch_count
            total_loss += loss.item() * batch_count
            total_positive_similarity += np.sum(positive_cosine_similarity)
            total_negative_similarity += np.sum(negative_cosine_similarity)
            
            # Write anchor (sample) features to file
            for sample in range(batch_count):
                writer.write(json.dumps({"label": int(anchor['label'][sample].detach().to('cpu').numpy()),
                                         "method_name": anchor['method_name'][sample],
                                         "features": anchor_rep[sample].detach().to('cpu').numpy().tolist() }) + "\n")    

    logging.info(f'[Test metrics] loss: {total_loss/data_cnt} positive similarity: {total_positive_similarity/data_cnt}  negative similarity: {total_negative_similarity/data_cnt}')

if __name__ == "__main__":
    extract()