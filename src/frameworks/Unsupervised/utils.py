import torch
import config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from itertools import cycle
from matplotlib import ticker
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from validclust.indices import dunn
from collections import defaultdict
from sklearn.metrics import pairwise_distances


sns.set_style('white')

classes = {
    0: "train",
    1: "save",
    2: "process",
    3: "forward",
    4: "predict"
}
    
def compute_clusters(model, dataloader, epoch_num=0, batch_num=0):

    model.eval()
    # Compute features without masking tokens when obtaining new clusters at end of epoch
    dataloader.mlm = False
    features, method_names = [], []
    with torch.no_grad():
        for _, sample in tqdm(enumerate(dataloader, 1), desc=f"Computing clusters", total=dataloader.__len__()):

            input_ids = sample['input_ids'].to(config.DEVICE)
            token_type_ids = sample['token_type_ids'].to(config.DEVICE)
            attention_mask = sample['attention_mask'].to(config.DEVICE)

            output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

            # Obtain a representation of the sample through mean pooling of the token features in the last hidden layer
            pooled_feats = output.hidden_states[-1].mean(dim=1).detach().cpu().numpy()

            for b in range(pooled_feats.shape[0]):
                features.append(pooled_feats[b, :])
                method_names.append(sample['method_name'][b])

                torch.cuda.empty_cache()
    
    # Turn mlm back on again for next epoch
    dataloader.mlm = True   

    # Perform k-means
    estimator = KMeans(n_clusters=config.NUM_CLUSTERS)
    clusters = estimator.fit(features)

    # Get labels
    cluster_labels = clusters.labels_
    true_labels = np.array([get_one_hot_label(n, classes) for n in method_names])

    # Use t-SNE and plot the clusters
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    tsne_result = tsne.fit_transform(features)

    # Plot and save computed clusters
    calculate_metrics(features, cluster_labels, true_labels, epoch_num, batch_num, ds='test')
    plot_clusters(tsne_result, cluster_labels, true_labels, epoch_num, batch_num, ds='test')

def plot_clusters(features, cluster_labels, true_labels, epoch=0, batch=0, ds='train'):
            
    k = len(np.unique(cluster_labels))
    # Visualize clusters with tSNE
    fig = plt.figure(figsize=(14,5))
    ax1 = fig.add_subplot(121)
    ax1.set_title(f'Kmeans Clusters (k={k})')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, k), colors):
        Xk = features[cluster_labels == klass]
        ax1.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.3)

    ax2 = fig.add_subplot(122)
    ax2.set_title('True labels')
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for klass, color in zip(range(0, len(classes)), colors):
        Xk = features[true_labels == klass]
        ax2.scatter(Xk[:, 0], Xk[:, 1],  c=color, alpha=0.3, label=classes[klass])
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{config.DATASET_DIR}/{ds}_clusters_epoch_{epoch}_batch_{batch}.pdf")
    


def calculate_metrics(features, predicted_labels, true_labels, epoch, batch, ds):
    
    k = len(np.unique(predicted_labels))
    distances = pairwise_distances(features)

    results = defaultdict(list)
    results['homogeneity_score'].append(metrics.homogeneity_score(true_labels, predicted_labels))
    results['completeness_score'].append(metrics.completeness_score(true_labels, predicted_labels))
    results['v_measure_score'].append(metrics.v_measure_score(true_labels, predicted_labels))
    results['adjusted_rand_score'].append(metrics.adjusted_rand_score(true_labels, predicted_labels))
    results['adjusted_mutual_info_score'].append(metrics.adjusted_mutual_info_score(true_labels, predicted_labels))
    results['average_jaccard_score'].append(np.mean(metrics.jaccard_score(true_labels, predicted_labels, average=None)))
    results['dunn_index'].append(dunn(distances, predicted_labels))
    if len(np.unique(predicted_labels)) == 1 or k == len(features):
        results['silhouette_score'].append(-1)
    else:
        results['silhouette_score'].append(metrics.silhouette_score(features, predicted_labels, metric='sqeuclidean'))
    
    if ds == 'test':
        latex_table = pd.DataFrame.from_dict(results).to_latex(index=False, float_format="%.3f").replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline')
        with open(f"./logs/tables/cluster_metrics_epoch_{epoch}_batch_{batch}.tex", "w") as writer:
            writer.write(latex_table)
    else:
        print(pd.DataFrame.from_dict(results))
            
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
        
def get_one_hot_label(method_name: str, classes: dict):
    matches = [x in method_name for x in classes.values()]
    if sum(matches) == 1:
        idx = np.where(matches)[0][0]
        return idx
    else:
        return 5