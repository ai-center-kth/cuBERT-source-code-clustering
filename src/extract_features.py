import os
import json
import torch
import config
import logging
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import ticker
from torch.cuda.amp import autocast
from dataset import TripletDataset, data_collator
from transformers import BertModel, BertConfig
from tokenizer.huggingface_compatible_tokenizer import CuBertHugTokenizer
from sklearn.metrics.pairwise import paired_cosine_distances, euclidean_distances

logging.getLogger().setLevel(logging.INFO)

def extract():

    # Create log directory
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    
    # Load pretrained model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertModel.from_pretrained(pretrained_model_name_or_path=config.MODEL_CHECKPOINT_PATH, config=model_config).to('cuda')
    model.eval()

    logging.info(f'Loaded model on {config.DEVICE}')

    # Set up tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    test_ds = TripletDataset(f'{config.DATASET_DIR}/test.json', tokenizer)
    test_dataloader = torch.utils.data.DataLoader(test_ds, collate_fn=data_collator, batch_size = 1, num_workers = 2)

    # Set loss function
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y))


    total_loss, total_positive_similarity, total_negative_similarity, total_positive_euclidean_distance, total_negative_euclidean_distance, data_cnt = 0, 0, 0, 0, 0, 0
    with open(f'{config.LOG_DIR}/features.json', 'w') as writer:
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_dataloader, 1), desc="Evaluating", total=test_dataloader.__len__()):

                batch_count = np.shape(batch['anchor']['method_name'])[0]
                with autocast():
                    features, names = {}, {}
                    for key in batch.keys():
                        input_ids = batch[key]['input_ids'].to(config.DEVICE)
                        token_type_ids = batch[key]['token_type_ids'].to(config.DEVICE)
                        attention_mask = batch[key]['attention_mask'].to(config.DEVICE)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                        features[key] = outputs.hidden_states[-1][:,0,:]
                        names[key] = batch['anchor']['method_name']

                    # Compute loss
                    anchor_rep, positive_rep, negative_rep = features.values()
                    loss = loss_fn(anchor_rep, positive_rep, negative_rep)
            
                # Compute similarities within batch
                positive_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                          positive_rep.detach().to('cpu').numpy()))
                negative_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                          negative_rep.detach().to('cpu').numpy()))

                positive_euclidean_distance = euclidean_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                      positive_rep.detach().to('cpu').numpy())
                negative_euclidean_distance = euclidean_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                      negative_rep.detach().to('cpu').numpy())

                data_cnt += batch_count
                total_loss += loss.item() * batch_count
                total_positive_similarity += np.sum(positive_cosine_similarity)
                total_negative_similarity += np.sum(negative_cosine_similarity)
                total_positive_euclidean_distance += np.sum(positive_euclidean_distance)
                total_negative_euclidean_distance += np.sum(negative_euclidean_distance)

                # Write anchor (sample) features to file
                for feature, name in zip(features['anchor'], names['anchor']):
                    writer.write(json.dumps({"method_name": name,
                                             "features": feature.detach().to('cpu').numpy().tolist() }) + "\n")    

    logging.info(f'[Test metrics] loss: {total_loss / data_cnt} positive similarity: {total_positive_similarity / data_cnt} negative similarity: {total_negative_similarity / data_cnt} positive euclidean distance: {total_positive_euclidean_distance / data_cnt} negative euclidean distance: {total_negative_euclidean_distance / data_cnt}')

if __name__ == "__main__":
    extract()