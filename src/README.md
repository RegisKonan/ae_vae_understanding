# Source code

This folder contains the models, losses, datasets and training utilities used by the [experiment scripts](../script).

```
src/
├── models.py                 # AE and VAE architectures
├── main/
│   ├── datasets.py           # MNIST, FashionMNIST and FreyFace loaders
│   └── loss_function.py      # AE and VAE (ELBO) losses
└── utilities/
    ├── trained_models.py     # training, evaluation, k-fold cross-validation
    ├── plotting.py           # metric curves and evaluation tables
    └── latent_space_sample.py# latent-space plots and data generation
```

## Models: [`models.py`](models.py)

- **Autoencoder (`AE`):** a deterministic encoder maps an input $x \in \mathbb{R}^{D_x}$ to a latent code $z \in \mathbb{R}^{D_z}$, and a decoder reconstructs $x$ from $z$.
- **Variational Autoencoder (`VAE`):** the `Encoder` outputs the mean $\mu$ and log-variance $\log \sigma^2$ of $q(z \mid x)$. A latent sample is drawn with the **reparameterization trick**, $z = \mu + \sigma \odot \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, I)$, and the `Decoder` maps $z$ back to the data space. Several samples ($L$) can be drawn per input.

### Architecture with a variable number of layers

The number of features decreases linearly from the input dimension $D_x$ to the latent dimension $D_z$ over $n_L$ layers:

$$
D_{x_i} = D_x - \left\lfloor \frac{(D_x - D_z)\, i}{n_L} \right\rfloor, \qquad i = 0, 1, \dots, n_L
$$

where $D_{x_i}$ is the number of features at layer $i$. The decoder mirrors the encoder. This makes it possible to change the depth with a single parameter, `n_layers`.

## Losses: [`main/loss_function.py`](main/loss_function.py)

- **AE:** reconstruction loss, either binary cross-entropy (BCE) or mean squared error (MSE).
- **VAE:** negative ELBO = reconstruction loss + $\beta \times$ KL divergence. Setting $\beta = 1$ gives the standard VAE, and $\beta \neq 1$ gives a β-VAE. Two priors are supported through the `kl_div` argument:

| `kl_div` | Prior $p(z)$ |
|---|---|
| `standard_gaussian` | $\mathcal{N}(0, I)$ |
| `mean_conditional_gaussian` | $\mathcal{N}(\mu_2, \mathrm{diag}(\sigma_2^2))$, with `mean_2` and `var_2` set by the user |

## Datasets: [`main/datasets.py`](main/datasets.py)

- **MNIST** and **FashionMNIST:** downloaded automatically with `torchvision`.
- **FreyFace:** loaded from [`data/FreyFace/frey_rawface.mat`](../data/FreyFace).

## Utilities

| File | Content |
|---|---|
| [`utilities/trained_models.py`](utilities/trained_models.py) | Training and evaluation loops for AEs and VAEs, k-fold cross-validation, visualisation of original and reconstructed images |
| [`utilities/plotting.py`](utilities/plotting.py) | Plots of BCE, MSE, L1 and ELBO curves (mean ± std over folds), and evaluation tables |
| [`utilities/latent_space_sample.py`](utilities/latent_space_sample.py) | Latent-space visualisation and generation of new data by sampling the latent space |

## Summary of findings

- **AE vs VAE:** AEs reconstruct images better, especially for larger $D_z$. VAEs are much better at **generating new data**, thanks to their regularised latent space.
- **β-VAE:** as $\beta \to 0$, the VAE behaves like an AE (good reconstructions, less regular latent space). Larger values of β give a smoother, more clustered latent space and better generation, at the cost of reconstruction detail.

See the [Master's project report](../reports/Regis_Djaha_AMMI_project_2023.pdf) for the full analysis.
