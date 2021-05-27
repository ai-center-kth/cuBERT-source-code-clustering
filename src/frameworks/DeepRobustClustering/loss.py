import torch

class DRCLoss(torch.nn.Module):
    """
    Implementation of the loss function described in Deep Robust Clustering By Contrastive Learning
    https://arxiv.org/pdf/2008.03030.pdf
    """
    def __init__(self, lmbda, temperature_af, temperature_ap):
        super(DRCLoss, self).__init__()
        self.lmbda = lmbda
        self.temperature_af = temperature_af
        self.temperature_ap = temperature_ap

    def forward(self, ap1, af1, ap2, af2):
        
        # N - Number of samples in batch, K - Dimensionality of features/probabilities (i.e. num clusters)
        N, K = ap1.shape 
        
        # Todo: Tidy up this ugly into matrix operations now that debugging is done
        af_loss = 0
        for i in range(N):
            numerator = torch.exp(torch.dot(af1[i,:], af2[i,:]) / self.temperature_af)
            denom = 0
            for j in range(N):
                denom += torch.exp(torch.dot(af1[i,:], af2[j,:]) / self.temperature_af)
                
            af_loss -= torch.log(numerator/denom) / N

            
        ap_loss = 0
        for i in range(K):
            numerator = torch.exp(torch.dot(ap1[:, i], ap2[:, i]) / self.temperature_ap)
            
            denom = 0
            for j in range(K):
                denom += torch.exp(torch.dot(ap1[:,i], ap2[:,j]) / self.temperature_ap)
        
            ap_loss -= torch.log(numerator/denom) / K
                    
        regularization_loss = 0
        for i in range(K):               
            regularization_loss += torch.square(ap1[:,i].sum()) / N
            
        return af_loss + ap_loss + self.lmbda * regularization_loss, af_loss, ap_loss, regularization_loss