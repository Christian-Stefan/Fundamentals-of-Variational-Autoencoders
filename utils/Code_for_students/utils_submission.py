from utils.Code_for_students.imports_submission import *

@staticmethod
def mini_Pipeline(batch_size, data_loc):
    train, test, val = create_dataloaders(data_loc=data_loc, batch_size=batch_size, transform=None)

    return train, test, val

from torch.nn import functional as F
def vae_loss(reconstructed_x, original_x, mean, log_std, beta=None):
    """
    Motivation: The learning experience seemed to be more rewarding when building myself the VAE Loss rather than using a built-in torch function.
    
    Definition: Combines Mean Squared Error (MSE) with KL Divergence.
    :param None|float beta: is used to balance the two terms. However, by default we assume the two underlying loss components to be equally important.
    """
    if beta is not None:
        # 1. Defining the first term
        # Term 1: Reconstruction Loss (How well did it copy the image?)
        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='mean')
        
        # 2. Defining the second term
        # Term 2: KL Divergence (How close is the latent space to a normal distribution/(our prior)?)
        # Adhering to the fomrmula presented in slide lectures: sum( µ^2 - 2log(σ) + σ^2 - 1 )
        # µ^2         -> mean.pow(2)
        # 2log(σ)     -> 2 * log_std
        # σ^2         -> torch.exp(2 * log_std)
        kl_divergence = torch.sum(mean.pow(2) - (2 * log_std) + torch.exp(2 * log_std) - 1)

        # 3.1 Normalize KL by batch size so it scales nicely with MSE
        kl_divergence = kl_divergence / original_x.size(0)
        # 3.2 Total Loss = Reconstruction + (beta * KL)
        total_loss = recon_loss + beta * kl_divergence
    else:
        recon_loss = F.mse_loss(reconstructed_x, original_x, reduction='sum')/original_x.size(0)
        kl_divergence = -0.5 * torch.sum(1+2*log_std-mean.pow(2)-torch.exp(2*log_std))
        kl_divergence = kl_divergence / original_x.size(0)
        total_loss = recon_loss + kl_divergence
        
    return total_loss, recon_loss, kl_divergence