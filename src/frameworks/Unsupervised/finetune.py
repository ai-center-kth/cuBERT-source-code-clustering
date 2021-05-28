import os
import torch
import config
import logging
import numpy as np


from tqdm import tqdm
from .utils import compute_clusters, save_history
from .loss import ClusteringLoss
from dataset import UnsupervisedDataset
from torch.cuda.amp import autocast, GradScaler
from transformers import BertForMaskedLM, BertConfig
from tokenizer.huggingface_compatible_tokenizer import CuBertHugTokenizer

logging.getLogger().setLevel(logging.INFO)
    
def unsupervised_finetune():
    
    # Create log directory
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Load pretrained model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path=config.MODEL_PATH, config=model_config).to('cuda')

    logging.info(f'Loaded model on {config.DEVICE}')

    # Set up tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    train_ds = UnsupervisedDataset(f"{config.DATASET_DIR}/train.json", tokenizer, mlm=True, mlm_probability=0.1)
    val_ds = UnsupervisedDataset(f"{config.DATASET_DIR}/val.json", tokenizer, mlm=True, mlm_probability=0.1)
    test_ds = UnsupervisedDataset(f"{config.DATASET_DIR}/test.json", tokenizer, mlm=True, mlm_probability=0.1)
        
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle=True, num_workers = 4)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size = config.BATCH_SIZE, num_workers = 4)
    test_dataloader = torch.utils.data.DataLoader(test_ds, batch_size = config.BATCH_SIZE, num_workers = 4)

    # Instantiate optimizer and grad scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler = GradScaler()
    
    # Instantiate loss function
    loss_fn = ClusteringLoss(model, train_dataloader)

    # Create dictionary for storing training metrics
    history = {'loss': {'eval': list(), 'train': list()},
              'mlm_loss': {'eval': list(), 'train': list()},
              'kl_loss': {'eval': list(), 'train': list()}}
    
    

    # Compute the initial clusters on the test set
    logging.info(f'Computing initial clusters on test set')   
    compute_clusters(model, test_dataloader)

    logging.info(f'Training model on {config.DEVICE}')
    for epoch in range(1, config.EPOCHS + 1):
            model = train_epoch(model, tokenizer, scaler, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, test_dataloader, epoch)
            if epoch < config.EPOCHS:
                # Re-compute the new clusters on the training set
                loss_fn.compute_centroids(epoch)
                # Compute the clusters on the test set
                compute_clusters(model, test_dataloader, epoch)
            

def train_epoch(model, tokenizer, scaler, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, test_dataloader, epoch_idx):

    total_loss, total_mlm_loss, total_kl_loss, data_cnt = 0, 0, 0, 0
    for batch_idx, sample in tqdm(enumerate(train_dataloader, 1),
                                                        desc=f"Training on epoch {epoch_idx}/{config.EPOCHS}",
                                                        total=train_dataloader.__len__()):
        model.train()
        optimizer.zero_grad()
        
        batch_count = np.shape(sample['labels'])[0]
        with autocast():
            # Get anchor features
            input_ids = sample['input_ids'].to(config.DEVICE)
            token_type_ids = sample['token_type_ids'].to(config.DEVICE)
            attention_mask = sample['attention_mask'].to(config.DEVICE)
            mlm_labels = sample['labels'].to(config.DEVICE)

            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=mlm_labels)
            
            # Get MLM loss
            mlm_loss = output.loss
            
            # Obtain a representation of the sample through mean pooling of the token features in the last hidden layer
            features = output.hidden_states[-1].mean(dim=1, dtype=torch.float32).detach().cpu()

            # Compute clustering loss
            clustering_loss, clustering_labels = loss_fn(features)
            
            # Total loss
            loss = mlm_loss + clustering_loss
            
            
        data_cnt += batch_count
        total_loss += loss.item() * batch_count
        total_mlm_loss += mlm_loss.item() * batch_count
        total_kl_loss += clustering_loss.item() * batch_count
        
        # Update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()
        
        if (batch_idx % config.NUM_BATCHES_UNTIL_LOG == 0 or batch_idx == train_dataloader.__len__()):
            logging.info(f'\n[epoch {epoch_idx} - batch {batch_idx} - train] loss: {total_loss / data_cnt} mlm-loss: {total_mlm_loss / data_cnt} KL-loss: {total_kl_loss / data_cnt}')

        if (batch_idx % config.NUM_BATCHES_UNTIL_EVAL == 0 or batch_idx == train_dataloader.__len__()):
            history['loss']['train'].append(total_loss / data_cnt)
            history['mlm_loss']['train'].append(total_mlm_loss / data_cnt)
            history['kl_loss']['train'].append(total_kl_loss / data_cnt)

            val_loss, val_mlm_loss, val_kl_loss = evaluate(model, loss_fn, val_dataloader)
            history['loss']['eval'].append(val_loss)
            history['mlm_loss']['eval'].append(val_mlm_loss)
            history['kl_loss']['eval'].append(val_kl_loss)

            save_history(history)
            logging.info(f'\n[epoch {epoch_idx} - batch {batch_idx} - val] loss: {val_loss} mlm-loss: {val_mlm_loss} KL-loss: {val_kl_loss}')
            
            compute_clusters(model, test_dataloader, epoch_idx, batch_idx)
            
            # Save if no previous history, or if new result is better than previous
            if len(history['loss']['eval']) == 0 or val_loss <= min(history['loss']['eval'][:-1], default=val_loss):
                    torch.save(model.state_dict(), f'./model/checkpoints/scubert.ckpt')
                    logging.info(f'[epoch {epoch_idx} - batch {batch_idx}] model saved')

            scheduler.step(val_loss)
    
    return model
                

def evaluate(model, loss_fn, dataloader):
    
    model.eval()
    logging.info('Evaluating...')
    
    total_loss, total_mlm_loss, total_kl_loss, data_cnt = 0, 0, 0, 0
    with torch.no_grad():
        for batch_idx, sample in tqdm(enumerate(dataloader, 1), desc="Evaluating", total=dataloader.__len__(), disable=True):
            
            batch_count = np.shape(sample['labels'])[0]
            with autocast():
                # Get anchor features
                input_ids = sample['input_ids'].to(config.DEVICE)
                token_type_ids = sample['token_type_ids'].to(config.DEVICE)
                attention_mask = sample['attention_mask'].to(config.DEVICE)
                mlm_labels = sample['labels'].to(config.DEVICE)

                output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=mlm_labels)

                # Get MLM loss
                mlm_loss = output.loss

                # Obtain a representation of the sample through mean pooling of the token features in the last hidden layer
                features = output.hidden_states[-1].mean(dim=1).detach().cpu()

                # Compute clustering loss
                clustering_loss, clustering_labels = loss_fn(features)

                # Total loss
                loss = mlm_loss + clustering_loss

        
            data_cnt += batch_count
            total_loss += loss.item() * batch_count
            total_mlm_loss += mlm_loss.item() * batch_count
            total_kl_loss += clustering_loss.item() * batch_count

    return total_loss / data_cnt, total_mlm_loss / data_cnt, total_kl_loss / data_cnt