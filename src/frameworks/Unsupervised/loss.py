import torch
import config

from tqdm import tqdm
from transformers import BertForMaskedLM
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from .utils import plot_clusters

class ClusteringLoss(torch.nn.Module):
    """
    A loss function based on the MLM and KL-divergence losses as presented in `Unsupervised Fine-tuning for Text Clustering`
    https://www.aclweb.org/anthology/2020.coling-main.482.pdf
    """
    def __init__(self, model: BertForMaskedLM, dataloader: torch.utils.data.DataLoader, alpha: float = 1.0):
        super(ClusteringLoss, self).__init__()
        self.model = model
        self.dataloader = dataloader
        self.alpha = alpha
        self.centroids = torch.empty(size=(config.NUM_CLUSTERS, model.config.hidden_size))
        # Init by computing initial centroids
        self.compute_centroids()
    
    def compute_soft_assignment(self, x):
        return (1 + (x**2)/self.alpha) ** (-(self.alpha + 1) / 2)
    
    def forward(self, features):

        # Compute distance between each feature and cluster centroid
        distances = torch.cdist(features, self.centroids)
        
        # Compute similarity matrix (distribution)
        soft_assign = distances.apply_(self.compute_soft_assignment)
        Q = soft_assign / soft_assign.sum(axis=1, keepdim=True)
        
        # Compute target (distribution)
        soft_cluster_freq = Q.sum(axis=0, keepdim=True)
        targ = Q.apply_(lambda x: x ** 2) / soft_cluster_freq
        P = targ / targ.sum(axis=1, keepdim=True)
        
        loss = (P * (P / Q).log()).sum(axis=1).sum()
        labels = torch.argmax(Q, dim=1)

        return loss, labels

    def compute_centroids(self, epoch=0):
        
        self.model.eval()
        # Compute features without masking tokens when obtaining new clusters at end of epoch
        self.dataloader.mlm = False
        features, method_names = [], []
        with torch.no_grad():
            for batch_idx, sample in tqdm(enumerate(self.dataloader, 1), desc=f"Computing clusters", total=self.dataloader.__len__()):

                input_ids = sample['input_ids'].to(config.DEVICE)
                token_type_ids = sample['token_type_ids'].to(config.DEVICE)
                attention_mask = sample['attention_mask'].to(config.DEVICE)

                output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

                # Obtain a representation of the sample through mean pooling of the token features in the last hidden layer
                pooled_feats = output.hidden_states[-1].mean(dim=1).detach().cpu().numpy()

                for b in range(pooled_feats.shape[0]):
                    features.append(pooled_feats[b, :])
                    method_names.append(sample['method_name'][b])


                torch.cuda.empty_cache()
        # Turn mlm back on again for next epoch
        self.dataloader.mlm = True   
        
        # Perform k-means
        estimator = KMeans(n_clusters=config.NUM_CLUSTERS)
        clusters = estimator.fit(features)

        # Store new centroids
        self.centroids = torch.tensor(clusters.cluster_centers_, dtype=torch.float32)
    
        # Use t-SNE and plot the clusters
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=3000)
        tsne_result = tsne.fit_transform(features)
        
        # Plot and save computed clusters
        plot_clusters(tsne_result, method_names, clusters.labels_, epoch, ds='train')