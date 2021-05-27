import torch
import config
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import ticker
from transformers import BertConfig
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics.pairwise import paired_cosine_distances, euclidean_distances

from .loss import DRCLoss
from .model import BertForClassification
from .utils import compute_clusters
from dataset import DRCDataset, data_collator
from tokenizer.huggingface_compatible_tokenizer import CuBertHugTokenizer


logging.getLogger().setLevel(logging.INFO)


def drc_finetune():

    # Load pretrained model
    model_config = BertConfig.from_json_file(config.MODEL_CONFIG)
    model = BertForClassification(config.MODEL_PATH, model_config, config.NUM_CLUSTERS).to(config.DEVICE)
    
    logging.info(f"Loaded model on {config.DEVICE}")

    # Initiaize tokenizer
    tokenizer = CuBertHugTokenizer(config.MODEL_VOCAB)

    # Configure data loaders
    train_ds = DRCDataset(f"{config.DATASET_DIR}/train.json", tokenizer)
    val_ds = DRCDataset(f"{config.DATASET_DIR}/val.json", tokenizer)
    test_ds = DRCDataset(f"{config.DATASET_DIR}/test.json", tokenizer)

    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=4
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        prefetch_factor=4
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_ds,
        collate_fn=data_collator,
        batch_size=config.BATCH_SIZE,
        num_workers=4,
        prefetch_factor=4
    )
    # Instantiate optimizer and grad scaler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    scaler = GradScaler()

    # Instantiate loss function
    loss_fn = DRCLoss(config.LAMBDA, config.TEMPERATURE_AF, config.TEMPERATURE_AP)
    
    # Instantiate dictionary for storing training evaluation metrics
    history = {
        "loss": {"eval": list(), "train": list()},
        "cosine_similarity": {"eval": list(), "train": list()},
        "euclidean_distance": {"eval": list(), "train": list()},
    }

    logging.info(f"Training model on {config.DEVICE}")

    compute_clusters(model, test_dataloader, ds_name='test')
    for epoch in range(1, config.EPOCHS + 1):
        train_epoch(
            model,
            scaler,
            optimizer,
            scheduler,
            loss_fn,
            history,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            epoch,
        )
        if epoch < config.EPOCHS:
            compute_clusters(model, test_dataloader, epoch, ds_name='test')


def train_epoch(model, scaler, optimizer, scheduler, loss_fn,
                history, train_dataloader, val_dataloader, test_dataloader, epoch_idx):


    total_loss, total_af_loss, total_ap_loss, total_reg_loss, total_cosine_similarity, total_euclidean_distance, data_cnt = (0, 0, 0, 0, 0, 0, 0)
    
    for batch_idx, batch in tqdm(enumerate(train_dataloader, 1), desc=f"Training on epoch {epoch_idx}/{config.EPOCHS}", total=train_dataloader.__len__()):
        
        model.train()
        optimizer.zero_grad()
        batch_count = np.shape(batch["anchor"]["method_name"])[0]
        with autocast():
            assignment_features, assignment_probabilities = {}, {}
            for key in batch.keys():
                input_ids = batch[key]["input_ids"].to(config.DEVICE)
                token_type_ids = batch[key]["token_type_ids"].to(config.DEVICE)
                attention_mask = batch[key]["attention_mask"].to(config.DEVICE)
                
                probabilities, features = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                # Get pooled features
                assignment_features[key] = features
                # Get assignment
                assignment_probabilities[key] = probabilities

            # Compute loss
            ap1, ap2 = assignment_probabilities.values()
            af1, af2 = assignment_features.values()

            loss, af_loss, ap_loss, reg_loss = loss_fn(ap1, af1, ap2, af2)

        cosine_similarity = 1 - (
            paired_cosine_distances(
                af1.detach().to("cpu").numpy(), af2.detach().to("cpu").numpy()
            )
        )

        euclidean_distance = euclidean_distances(
            af1.detach().to("cpu").numpy(), af2.detach().to("cpu").numpy()
        )

        data_cnt += batch_count
        total_loss += loss.item() * batch_count
        total_af_loss += af_loss.item() * batch_count
        total_ap_loss += ap_loss.item() * batch_count
        total_reg_loss += reg_loss.item() * batch_count
        
        total_cosine_similarity += np.sum(cosine_similarity)
        total_euclidean_distance += np.sum(euclidean_distance)

        # Update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.empty_cache()

        # Check if logging should be done
        if ( batch_idx % config.NUM_BATCHES_TO_LOG == 0 or batch_idx == train_dataloader.__len__()):
            logging.info(
                f"\n[epoch {epoch_idx} - batch {batch_idx} - train] loss: {total_loss / data_cnt} AF loss: {total_af_loss / data_cnt} AP loss: {total_ap_loss / data_cnt} CR loss {total_reg_loss / data_cnt} cosine similarity: {total_cosine_similarity / data_cnt} euclidean distance: {total_euclidean_distance / data_cnt}"
            )

        # Check if evaluation should be done
        if (batch_idx % config.NUM_BATCHES_UNTIL_EVAL == 0 or batch_idx == train_dataloader.__len__()):
            history["loss"]["train"].append(total_loss / data_cnt)
            history["cosine_similarity"]["train"].append(
                total_cosine_similarity / data_cnt
            )
            history["euclidean_distance"]["train"].append(
                total_euclidean_distance / data_cnt
            )

            # Validate the model at the current step
            val_loss, val_af_loss, val_ap_loss, val_reg_loss, avg_pos_sim, avg_pos_euc_dist = evaluate(
                model, loss_fn, val_dataloader)

            history["loss"]["eval"].append(val_loss)
            history["cosine_similarity"]["eval"].append(avg_pos_sim)
            history["euclidean_distance"]["eval"].append(avg_pos_euc_dist)

            save_history(history)
            logging.info(
                f"\n[epoch {epoch_idx} - batch {batch_idx} - val] loss: {val_loss} AF loss: {val_af_loss} AP loss: {val_ap_loss} CR loss {val_reg_loss} cosine similarity: {avg_pos_sim} euclidean distance: {avg_pos_euc_dist}"
            )
            
            # Compute the resulting clusters on the test set
            compute_clusters(model, test_dataloader, epoch_idx, batch_idx, ds_name='test')

            # Save if no previous history, or if new result is better than previous
            if len(history["loss"]["eval"]) == 0 or val_loss <= min(history["loss"]["eval"][:-1], default=val_loss):
                torch.save(model.state_dict(), config.MODEL_CHECKPOINT_PATH)
                logging.info(f"[epoch {epoch_idx} - batch {batch_idx}] model saved")

            scheduler.step(val_loss)


def evaluate(model, loss_fn, dataloader):

    model.eval()
    logging.info("Evaluating...")

    total_loss, total_af_loss, total_ap_loss, total_reg_loss, total_cosine_similarity, total_euclidean_distance, data_cnt = (0, 0, 0, 0, 0, 0, 0)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader, 1), desc="Evaluating", total=dataloader.__len__(),disable=True):

            batch_count = np.shape(batch["anchor"]["method_name"])[0]
            with autocast():
                assignment_features, assignment_probabilities = {}, {}
                for key in batch.keys():
                    input_ids = batch[key]["input_ids"].to(config.DEVICE)
                    token_type_ids = batch[key]["token_type_ids"].to(config.DEVICE)
                    attention_mask = batch[key]["attention_mask"].to(config.DEVICE)

                    probabilities, features = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                    # Get pooled features
                    assignment_features[key] = features
                    # Get assignment
                    assignment_probabilities[key] = probabilities

                # Compute loss
                ap1, ap2 = assignment_probabilities.values()
                af1, af2 = assignment_features.values()

                loss, af_loss, ap_loss, reg_loss = loss_fn(ap1, af1, ap2, af2)

            # Compute similarities within batch
            cosine_similarity = 1 - (
                paired_cosine_distances(
                    af1.detach().to("cpu").numpy(), af2.detach().to("cpu").numpy()
                )
            )

            euclidean_distance = euclidean_distances(
                af1.detach().to("cpu").numpy(), af2.detach().to("cpu").numpy()
            )

            data_cnt += batch_count
            total_loss += loss.item() * batch_count
            total_af_loss += af_loss.item() * batch_count
            total_ap_loss += ap_loss.item() * batch_count
            total_reg_loss += reg_loss.item() * batch_count

            total_cosine_similarity += np.sum(cosine_similarity)
            total_euclidean_distance += np.sum(euclidean_distance)

    return (
        total_loss / data_cnt,
        total_af_loss / data_cnt,
        total_ap_loss / data_cnt,
        total_reg_loss / data_cnt,
        total_cosine_similarity / data_cnt,
        total_euclidean_distance / data_cnt,
    )


def save_history(history):
    for metric, data in history.items():
        # save raw data
        with open(f"{config.LOG_DIR}/{metric}.data", mode="w") as f:
            for name in data:
                datapoints = ",".join([str(v) for v in data[name]])
                f.write(f"{name},{datapoints}\n")

        # save graph
        train_metric = history[metric]["train"]
        eval_metric = history[metric]["eval"]

        x = np.linspace(
            1,
            len(train_metric) * config.NUM_BATCHES_UNTIL_EVAL * config.BATCH_SIZE,
            len(train_metric),
        )
        plt.figure()
        plt.plot(x, train_metric, marker="o", label=f"train_{metric}")
        plt.plot(x, eval_metric, marker="*", label=f"eval_{metric}")
        plt.title(f"{metric} for training and validation sets")
        plt.xlabel("Samples")
        plt.ylabel(f"{metric}")
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.legend()
        plt.savefig(f"{config.LOG_DIR}/{metric}.pdf")
        plt.close()