# Models

## Autoencoders (AEs) and VAriational Autoencoders (VAEs)

#### Description 


Autoencoders and Variational Autoencoders are powerful tools in machine learning for various applications such as image generation, anomaly detection, and  dimensionality reduction.

Autoencoder is used to learn efficient embeddings of unlabeled data that contain as much information as possible to then reconstruct as best as possible the original input based only on the learned embedding. The entire architecture of AE is deterministic. Unlike AEs, VAEs introduce a stochastic element to the encoding process. The latent variable $z$ is treated as a random variable in $R^{D_z}$.

### Architectures

In the context of dynamic Autoencoders (AEs) and Variationals Autoencoders (VAEs), various methodologies exist for their implementation. Specifically, within our approach, the dynamism is encapsulated by equation. This equation establishes a relationship that dynamically tunes the number of features $D_{x_i}$ at each layer of a neural network. It serves as a key component in our implementation, facilitating adaptive adjustments to the network architecture based on the defined variables.

Consider the equation:

$$ D_{x_i} = D_x - \left[ \frac{(D_x- D_{z}) \cdot i}{n_{layers}}\right] \quad \text{for} \quad i = 0, 1, \cdots n_{layers}$$

where

- $D_{x_i}$ represents the number of features at layer $i$
- $D_{x}$ is the initial number of features
- $D_{z}$ is the dimension of latent space or variable
- $i$ is the layer index, ranging from 0 to $n_{layers}$, and
- $n_{layers}$ is the total number of layers.

### Experiments

In this study, we explore the theories and carry out several experiments on AEs and VAEs.



The first experiment consists of a comparative analysis between AEs and VAEs. We explore two aspects: firstly, keeping the dimensions fixed and changing the number of layers, and secondly, keeping the number of layers constant while varying the dimensions. The samples resulting from this analysis are stored in files [AE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/c05677fd700c947ddbcc2038bb82f5640a97eb8c/results/AE_experiment_FashionMNIST) for AEs and [VAE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/c05677fd700c947ddbcc2038bb82f5640a97eb8c/results/VAE_experiment_FreyFace) for VAEs. Our results show that autoencoders perform very well in image reconstruction compared with variational autoencoders, this being more visible for a larger value of the D_z dimension (dimension of the latent space). Conversely, variational autoencoders are much more powerful for data generation compared with AEs.

The next experiment focuses on examining the impact of beta-VAE (see [beta-VAE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/c05677fd700c947ddbcc2038bb82f5640a97eb8c/results/beta-VAE_experiment_MNIST)). When beta approaches 0, it reflects the behaviour of autoencoders, while values closer to 1 resemble variational autoencoders. This experience confirms the theory. In particular, beta = 1 highlights better clustering compared to autoencoders, with this effect being more visible at higher beta values. In addition, beta VAEs with values greater than 1 are more powerful at data generation than traditional VAEs.


# Main 

The [main](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/main) contains the [datasets.py](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/main/datasets.py) and [function_loss.py](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/main/loss_function.py). Our datasets include MNIST and fashion MNIST data. We use BCE or MSE metrics as reconstruction losses for the AE model and ELBO for the VAE model. ELBO has two components: the reconstruction loss, which is the BCE or MSE metric, and the KL divergence loss.


## utilities

The [trained_models.py](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/utilities/trained_models.py)   contain training functions, evaluation functions, functions to visualize original and reconstructed data, and cross-validation functions for AE and VAE models.
[plotting.py](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/utilities/plotting.py) contains the function for plotting BCE, MSE, and L1 Loss metrics.

[latent_space_sample.py](https://github.com/RegisKonan/ae_vae_understanding/tree/5109887685e8c3e2e0b8385d874d231338b38644/src/utilities/latent_space_sample.py) contains the function for visualizing latent spaces and generating new data.
