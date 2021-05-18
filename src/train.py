import os
import torch
import config
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from matplotlib import ticker
from dataset import TripletDataset, data_collator
from transformers import BertModel, BertConfig
from tokenizer.huggingface_compatible_tokenizer import CuBertHugTokenizer
from sklearn.metrics.pairwise import paired_cosine_distances, euclidean_distances

sns.set_style('white')
logging.getLogger().setLevel(logging.INFO)

def train():
    
    # Create log directory
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Load pretrained model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertModel.from_pretrained(pretrained_model_name_or_path=config.MODEL_PATH, config=model_config).to('cuda')

    logging.info(f'Loaded model on {config.DEVICE}')

    # Set up tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    train_ds = TripletDataset(f'{config.DATASET_DIR}/train.json', tokenizer)
    val_ds = TripletDataset(f'{config.DATASET_DIR}/val.json', tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_ds, collate_fn=data_collator, batch_size = config.BATCH_SIZE, shuffle=True, num_workers = 4)
    val_dataloader = torch.utils.data.DataLoader(val_ds, collate_fn=data_collator, batch_size = config.BATCH_SIZE, num_workers = 4)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - torch.nn.functional.cosine_similarity(x, y))
    scaler = GradScaler()
    
    history = {'loss': {'eval': list(), 'train': list()},
               'positive_similarity': {'eval': list(), 'train': list()},
               'negative_similarity': {'eval': list(), 'train': list()},
               'positive_euclidean_distance': {'eval': list(), 'train': list()},
               'negative_euclidean_distance': {'eval': list(), 'train': list()},
    }
    
    logging.info(f'Training model on {config.DEVICE}')
    
    for epoch in range(1, config.EPOCHS + 1):
            model = train_epoch(model, tokenizer, scaler, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, epoch)
            

def train_epoch(model, tokenizer, scaler, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, epoch_idx):

    
    total_loss, total_positive_similarity, total_negative_similarity, total_positive_euclidean_distance, total_negative_euclidean_distance, data_cnt = 0, 0, 0, 0, 0, 0
    for batch_idx, batch in tqdm(enumerate(train_dataloader, 1),
                                                        desc=f"Training on epoch {epoch_idx}/{config.EPOCHS}",
                                                        total=train_dataloader.__len__()):
        model.train()
        optimizer.zero_grad()
        batch_count = np.shape(batch['anchor']['method_name'])[0]
        with autocast():
            features = {}
            for key in batch.keys():
                input_ids = batch[key]['input_ids'].to(config.DEVICE)
                token_type_ids = batch[key]['token_type_ids'].to(config.DEVICE)
                attention_mask = batch[key]['attention_mask'].to(config.DEVICE)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                features[key] = outputs.hidden_states[-1][:,0,:] # (num_layers, batch_size, sequence_length, hidden_size)

            # Compute loss
            anchor_rep, positive_rep, negative_rep = features.values()
            loss = loss_fn(anchor_rep, positive_rep, negative_rep)

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

        # Update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        
        if batch_idx % config.NUM_BATCHES_UNTIL_LOG == 0 or batch_idx == train_dataloader.__len__():
            logging.info(f'\n[epoch {epoch_idx} - batch {batch_idx} - train] loss: {total_loss / data_cnt} positive similarity: {total_positive_similarity / data_cnt} negative similarity: {total_negative_similarity / data_cnt} positive euclidean distance: {total_positive_euclidean_distance / data_cnt} negative euclidean distance: {total_negative_euclidean_distance / data_cnt}')

        if (batch_idx % config.NUM_BATCHES_UNTIL_EVAL == 0 or batch_idx == train_dataloader.__len__()):
            history['loss']['train'].append(total_loss / data_cnt)
            history['positive_similarity']['train'].append(total_positive_similarity / data_cnt)
            history['negative_similarity']['train'].append(total_negative_similarity / data_cnt)
            history['positive_euclidean_distance']['train'].append(total_positive_euclidean_distance / data_cnt)
            history['negative_euclidean_distance']['train'].append(total_negative_euclidean_distance / data_cnt)

            val_loss, avg_pos_sim, avg_neg_sim, avg_pos_euc_dist, avg_neg_euc_dist = evaluate(model, loss_fn, val_dataloader)
            history['loss']['eval'].append(val_loss)
            history['positive_similarity']['eval'].append(avg_pos_sim)
            history['negative_similarity']['eval'].append(avg_neg_sim)
            history['positive_euclidean_distance']['eval'].append(avg_pos_euc_dist)
            history['negative_euclidean_distance']['eval'].append(avg_neg_euc_dist)

            save_history(history)
            logging.info(f'\n[epoch {epoch_idx} - batch {batch_idx} - val] loss: {val_loss} positive similarity: {avg_pos_sim}  negative similarity: {avg_neg_sim} positive euclidean distance: {avg_pos_euc_dist} negative euclidean distance: {avg_neg_euc_dist}')

            # Save if no previous history, or if new result is better than previous
            if len(history['loss']['eval']) == 0 or val_loss <= min(history['loss']['eval'][:-1], default=val_loss):
                    torch.save(model.state_dict(), f'./model/checkpoints/scubert.ckpt')
                    logging.info(f'[epoch {epoch_idx} - batch {batch_idx}] model saved')

            scheduler.step(val_loss)
    
    return model
                

def evaluate(model, loss_fn, dataloader):
    
    model.eval()
    logging.info('Evaluating...')
    total_loss, total_positive_similarity, total_negative_similarity, total_positive_euclidean_distance, total_negative_euclidean_distance, data_cnt = 0, 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader, 1), desc="Evaluating", total=dataloader.__len__(), disable=True):
            
            batch_count = np.shape(batch['anchor']['method_name'])[0]
            with autocast():
                features = {}
                for key in batch.keys():
                    input_ids = batch[key]['input_ids'].to(config.DEVICE)
                    token_type_ids = batch[key]['token_type_ids'].to(config.DEVICE)
                    attention_mask = batch[key]['attention_mask'].to(config.DEVICE)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    features[key] = outputs.hidden_states[-1][:,0,:]

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

    return total_loss / data_cnt, total_positive_similarity / data_cnt, total_negative_similarity / data_cnt, total_positive_euclidean_distance / data_cnt, total_negative_euclidean_distance / data_cnt

def save_history(history):
    for metric, data in history.items():
        # save raw data
        with open(f'./logs/{metric}.data', mode='w') as f:
            for name in data:
                datapoints = ",".join([str(v) for v in data[name]])
                f.write(f"{name},{datapoints}\n")
        
        # save graph
        train_metric = history[metric]['train']
        eval_metric = history[metric]['eval']

        x = np.linspace(1, len(train_metric) * config.NUM_BATCHES_UNTIL_EVAL * config.BATCH_SIZE, len(train_metric))
        plt.figure()
        plt.plot(x, train_metric, marker='o', label=f'train_{metric}')
        plt.plot(x, eval_metric, marker='*', label=f'eval_{metric}')
        plt.title(f"{metric} for training and validation sets")
        plt.xlabel('Samples')
        plt.ylabel(f"{metric}")
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(f'./logs/{metric}.pdf')
        plt.close()

if __name__ == "__main__":
    train()