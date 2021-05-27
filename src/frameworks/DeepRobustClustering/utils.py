import config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from tqdm import tqdm
from itertools import cycle
from matplotlib import ticker
from sklearn.metrics.pairwise import euclidean_distances
from validclust.indices import dunn
from sklearn import metrics
from sklearn.manifold import TSNE
from collections import defaultdict

sns.set_style("white")

classes = {
    0: "train",
    1: "save",
    2: "process",
    3: "forward",
    4: "predict",
    5: "random",
}
    
def compute_clusters(model, dataloader, epoch_num=0, batch_num=0, ds_name='train'):

    model.eval()
    # Compute features without masking tokens when obtaining new clusters at end of epoch
    assignment_features, assignment_probabilities, cluster_labels, method_names = (
        [],
        [],
        [],
        [],
    )

    for _, batch in tqdm(enumerate(dataloader, 1), desc=f"Computing clusters", total=dataloader.__len__()):
                        
        batch_size = np.shape(batch["anchor"]["method_name"])[0]
        input_ids = batch["anchor"]["input_ids"].to(config.DEVICE)
        token_type_ids = batch["anchor"]["token_type_ids"].to(config.DEVICE)
        attention_mask = batch["anchor"]["attention_mask"].to(config.DEVICE)

        probabilities, features = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        probabilities = probabilities.detach().cpu().numpy()
        features = features.detach().cpu().numpy()

        for b in range(batch_size):
            if get_one_hot_label(batch["anchor"]["method_name"][b]) != 5:
                assignment_features.append(features[b, :])
                assignment_probabilities.append(probabilities[b, :])
                cluster_labels.append(np.argmax(probabilities[b, :]))
                method_names.append(batch["anchor"]["method_name"][b])
            
    # Compute t-SNE
    af_tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    af_tsne_result = af_tsne.fit_transform(assignment_features)

    # Compute t-SNE
    ap_tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
    ap_tsne_result = ap_tsne.fit_transform(assignment_probabilities)
    
    # Get the labels
    cluster_labels = np.array(cluster_labels)
    true_labels = np.array([get_one_hot_label(n) for n in method_names])
    
    # Plot and save the computed clusters
    plot_clusters(af_tsne_result, ap_tsne_result, cluster_labels, true_labels, epoch_num, batch_num, ds_name)
    calculate_metrics(assignment_probabilities, cluster_labels, true_labels, epoch_num, batch_num, ds_name)


def calculate_metrics(features, predicted_labels, true_labels, epoch, batch, ds):
    
    features = np.array(features)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    
    distances = euclidean_distances(X=features)
    k = len(np.unique(predicted_labels))
    results = defaultdict(list)
    results['homogeneity_score'].append(metrics.homogeneity_score(true_labels, predicted_labels))
    results['completeness_score'].append(metrics.completeness_score(true_labels, predicted_labels))
    results['v_measure_score'].append(metrics.v_measure_score(true_labels, predicted_labels))
    results['adjusted_rand_score'].append(metrics.adjusted_rand_score(true_labels, predicted_labels))
    results['adjusted_mutual_info_score'].append(metrics.adjusted_mutual_info_score(true_labels, predicted_labels))
    results['average_jaccard_score'].append(np.mean(metrics.jaccard_score(true_labels, predicted_labels, average=None)))
    try:
        di = dunn(dist=distances, labels=predicted_labels)
    except ValueError:
        di = 0.0 # Value error gets thrown if there is only a single cluster
    results['dunn_index'].append(di)
    if len(np.unique(predicted_labels)) == 1 or k == len(features):
        results['silhouette_score'].append(-1)
    else:
        results['silhouette_score'].append(metrics.silhouette_score(features, predicted_labels, metric='sqeuclidean'))
    
    print(f"Internal Cluster Evaluation metrics on the {ds} set:\n{pd.DataFrame.from_dict(results)}")
    if ds == 'test':
        latex_table = pd.DataFrame.from_dict(results).to_latex(index=False, float_format="%.3f").replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline')
        with open(f"{config.LOG_DIR}/cluster_metrics_epoch_{epoch}_batch_{batch}.tex", "w") as writer:
            writer.write(latex_table)

        
def plot_clusters(features, probabilities, predicted_labels, true_labels, epoch=0, batch=0, ds='train'):
    
    # Visualize clusters with tSNE
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_title(f"Assigned Clusters (k={config.NUM_CLUSTERS})")
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for klass, color in zip(range(0, config.NUM_CLUSTERS), colors):
        Xk = features[predicted_labels == klass]
        ax1.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.3)

    ax2 = fig.add_subplot(122)
    ax2.set_title("True labels")
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for klass, color in zip(range(0, len(classes)), colors):
        Xk = features[true_labels == klass]
        ax2.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.3, label=classes[klass])
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/assignment_feature_clusters_epoch_{epoch}_batch_{batch}.pdf")

    plt.close()

    # Visualize clusters with tSNE
    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_title(f"Assigned Clusters (k={config.NUM_CLUSTERS})")
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for klass, color in zip(range(0, config.NUM_CLUSTERS), colors):
        Xk = probabilities[predicted_labels == klass]
        ax1.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.3)

    ax2 = fig.add_subplot(122)
    ax2.set_title("True labels")
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for klass, color in zip(range(0, len(classes)), colors):
        Xk = probabilities[true_labels == klass]
        ax2.scatter(Xk[:, 0], Xk[:, 1], c=color, alpha=0.3, label=classes[klass])
    ax2.legend()
    plt.tight_layout()
    plt.savefig(f"{config.LOG_DIR}/assignment_probability_clusters_epoch_{epoch}_batch_{batch}.pdf")
    plt.close()
    
    
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


def get_one_hot_label(method_name: str):
    matches = [x in method_name for x in classes.values()]
    if sum(matches) == 1:
        idx = np.where(matches)[0][0]
        return idx
    else:
        return 5