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

def train(epochs, lr=2e-5):
    
    # Load pretrained model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertForPreTraining.from_pretrained(pretrained_model_name_or_path=config.MODEL_PATH, from_tf=True, config=model_config).to('cuda')
    # print(model)

    logging.info(f'Loaded model on {config.DEVICE}')

    # Set up tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    train_ds = TripletDataset('./data/train.json')
    val_ds = TripletDataset('./data/val.json')

    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size = config.BATCH_SIZE, shuffle=True, num_workers = 2)
    val_dataloader = torch.utils.data.DataLoader(val_ds, batch_size = config.BATCH_SIZE, num_workers = 2)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    loss_fn = torch.nn.TripletMarginLoss(margin=1.0, p=2).to(config.DEVICE)
    
    history = {'loss': {'eval': list(), 'train': list()},
               'positive_similarity': {'eval': list(), 'train': list()},
               'negative_similarity': {'eval': list(), 'train': list()}
    }
    
    logging.info(f'Training model on {config.DEVICE}')
    
    for epoch in range(1, epochs + 1):
            model = train_epoch(model, tokenizer, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, epoch)
            

def train_epoch(model, tokenizer, optimizer, scheduler, loss_fn, history, train_dataloader, val_dataloader, epoch_idx):

    model.train()
    total_loss, total_positive_similarity, total_negative_similarity, data_cnt = 0, 0, 0, 0
    
    for batch_idx, (anchor, positive, negative) in tqdm(enumerate(train_dataloader, 1),
                                                        desc=f"Training on epoch {epoch_idx}/{config.EPOCHS}",
                                                        total=train_dataloader.__len__()):
        
        optimizer.zero_grad()
        batch_count = np.shape(anchor['method'])[0]

        # Get anchor features
        inputs = tokenizer(anchor['method'])
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        anchor_rep = outputs.hidden_states[-1][:,0,:] # (batch_size, sequence_length, hidden_size)


        # Get positive features
        inputs = tokenizer(positive['method'])
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        positive_rep = outputs.hidden_states[-1][:,0,:]

        # Get negative features
        inputs = tokenizer(negative['method'])
        input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
        attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        negative_rep = outputs.hidden_states[-1][:,0,:]

        # Compute loss
        loss = loss_fn(anchor_rep, positive_rep, negative_rep)

        positive_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                  positive_rep.detach().to('cpu').numpy()))
        negative_cosine_similarity = 1 - (paired_cosine_distances(anchor_rep.detach().to('cpu').numpy(),
                                                                  negative_rep.detach().to('cpu').numpy()))
        data_cnt += batch_count
        total_loss += loss.item() * batch_count
        total_positive_similarity += np.sum(positive_cosine_similarity)
        total_negative_similarity += np.sum(negative_cosine_similarity)

        # Update weights
        loss.backward()
        optimizer.step()

        if batch_idx % config.NUM_BATCHES_TO_LOG == 0 or batch_idx == train_dataloader.__len__():
            logging.info(f'[epoch {epoch_idx} - batch {batch_idx} - train] loss: {total_loss / data_cnt} positive similarity: {total_positive_similarity / data_cnt} negative similarity: {total_negative_similarity / data_cnt}')
        
        if (batch_idx % config.NUM_BATCHES_UNTIL_EVAL == 0 or batch_idx == train_dataloader.__len__()):
            history['loss']['train'].append(total_loss / data_cnt)
            history['positive_similarity']['train'].append(total_positive_similarity / data_cnt)
            history['negative_similarity']['train'].append(total_negative_similarity / data_cnt)
            
            val_loss, avg_pos_sim, avg_neg_sim = evaluate(model, tokenizer, history, loss_fn, val_dataloader, epoch_idx)
            history['loss']['eval'].append(val_loss)
            history['positive_similarity']['eval'].append(avg_pos_sim)
            history['negative_similarity']['eval'].append(avg_neg_sim)
            
            save_history(history)
            logging.info(f'[epoch {epoch_idx} - batch {batch_idx} - val] loss: {val_loss} positive similarity: {avg_pos_sim}  negative similarity: {avg_neg_sim}')
            
            # Save if no previous history, or if new result is better than previous
            if len(history['loss']['eval']) == 0 or val_loss <= min(history['loss']['eval'][:-1], default=val_loss):
                    torch.save(model.state_dict(), f'./model/checkpoints/scubert.ckpt')
                    logging.info(f'[epoch {epoch_idx} - batch {batch_idx}] model saved')
            scheduler.step(val_loss)
    
    return model
                

def evaluate(model, tokenizer, history, loss_fn, dataloader, epoch):
    
    model.eval()
    logging.info('Evaluating...')
    total_loss, total_positive_similarity, total_negative_similarity, data_cnt = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch_idx, (anchor, positive, negative) in tqdm(enumerate(dataloader, 1), desc="Evaluating", total=dataloader.__len__(), disable=False):
            
            batch_count = np.shape(anchor['method'])[0]

            # Get anchor features
            inputs = tokenizer(anchor['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            anchor_rep = outputs.hidden_states[-1][:,0,:]
            
            # Get positive features
            inputs = tokenizer(positive['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            positive_rep = outputs.hidden_states[-1][:,0,:] 

            # Get negative features
            inputs = tokenizer(negative['method'])
            input_ids = torch.tensor(inputs['input_ids'], dtype=torch.int, device=config.DEVICE)
            token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.int, device=config.DEVICE)
            attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.int, device=config.DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            negative_rep = outputs.hidden_states[-1][:,0,:] 

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

    return total_loss / data_cnt, total_positive_similarity / data_cnt, total_negative_similarity / data_cnt

def save_history(history):
    for metric, data in history.items():
        # save raw data
        with open(f'./logs/{metric}.data', mode='w') as f:
            for name in data:
                datapoints = ",".join([str(v) for v in data[name]])
                f.write(f"{name} {datapoints}\n")
        
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
        plt.savefig(f'./logs/{metric}.png')
        plt.close()

if __name__ == "__main__":
    train(config.EPOCHS)