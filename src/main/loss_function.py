import torch
import torch.nn.functional as F




#######  AE loss function #####

# AE criterion: reconstruction loss, based on Gaussian or Bernoulli likelihoods
def AE_criterion(x_out, x_in, AE_likelihood):

    if AE_likelihood == "gaussian":
      recon_loss = F.mse_loss(x_out, x_in, reduction='mean')
    elif AE_likelihood == "bernoulli":
      recon_loss = F.binary_cross_entropy(x_out, x_in, reduction='mean')

    else:
        raise ValueError("Invalid likelihood type. Supported types are 'gaussian' and 'bernoulli'.")

    loss = recon_loss

    return loss




#######  VAE ELBO #####

# VAE Criterion, as combination of reconstruction loss and KL divergence
# Note that the reconstruction loss can be based on the Gaussian or Bernoulli likelihood
# Note that the KL divergence term can be weighted by the input $\beta$ parameter
# def VAE_criterion(x_out, x_in, z_mu, z_logvar, kld_loss , mean_2, var_2, L, VAE_likelihood, beta=1.0):
def VAE_loss_function(x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta,kl_div, mean_2=None, var_2=None):
    recons_losses = []  # Store individual reconstruction losses

    if VAE_likelihood == "gaussian":
        for i in range(L):
            # Flatten x_in to match the shape of x_out
            x_in_flat = x_in.view(x_out[i::L].shape)
            # Calculate the mean squared error (MSE) loss for each sample
            recons_loss = F.mse_loss(x_out[i::L], x_in_flat, reduction='sum')
            recons_losses.append(recons_loss)
    elif VAE_likelihood == "bernoulli":
        for i in range(L):
            # Flatten x_in to match the shape of x_out
            x_in_flat = x_in.view(x_out[i::L].shape)
            # Calculate the binary cross-entropy loss for each sample
            recons_loss = F.binary_cross_entropy(x_out[i::L], x_in_flat, reduction='sum')
            recons_losses.append(recons_loss)
    else: 
        raise ValueError("Invalid KLD loss type. Supported types are 'gaussian' and 'bernoulli'.")  
    # Calculate the mean loss over the L samples
    recons_loss = (1 / L) * sum(recons_losses)/ x_out.size(0)
    
    if kl_div == "standard_gaussian" :
        
        kld_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())/ x_out.size(0)
        
        
    elif kl_div == "mean_conditional_gaussian" :
        
        if mean_2 is None or var_2 is None:
            
            raise ValueError("mean_2 and var_2 must be provided for 'mean_conditional_gaussian' KLD loss.")

        kld_loss = 0.5 * torch.sum(

            torch.pow(z_mu - mean_2, 2) / torch.exp(torch.log(var_2))

            + z_logvar.exp() / torch.exp(torch.log(var_2))

            - 1.0

            - z_logvar

            + torch.log(var_2)

        ) / x_out.size(0)
    
    else : 
        
        raise ValueError("Invalid KLD loss type. Supported types are 'standard_gaussian' and 'mean_conditional_gaussian'.")
    
    loss = (recons_loss + beta * kld_loss)
    

    return recons_loss, kld_loss, loss

def VAE_criterion(kl_div, x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta, mean_2=None, var_2=None):
    if kl_div == "standard_gaussian":
        return VAE_loss_function(x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta, kl_div)
    elif kl_div == "mean_conditional_gaussian":
        if mean_2 is None or var_2 is None:
            raise ValueError("mean_2 and var_2 must be provided for 'mean_conditional_gaussian' KLD loss.")
        return VAE_loss_function(x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta, kl_div, mean_2, var_2)
    else:
        raise ValueError("Invalid KLD loss type. Supported types are 'standard_gaussian' and 'mean_conditional_gaussian'.")

