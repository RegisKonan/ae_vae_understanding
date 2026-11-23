# Experiment scripts

This folder contains the scripts that **train** AEs and VAEs and **analyse** the trained models. All scripts must be run from the **root of the repository**, for example `python script/AE_VAE_experiment_nL.py …`.

The datasets (MNIST, FashionMNIST, FreyFace) are loaded by [`src/main/datasets.py`](../src/main/datasets.py). Every model is trained with **k-fold cross-validation**, and the results are saved in [`results/`](../results).

## 1. Training scripts

| Script | What it does | Example |
|---|---|---|
| [`AE_VAE_experiment_nL.py`](AE_VAE_experiment_nL.py) | Trains AEs or VAEs with a **fixed latent dimension $D_z$** and **several numbers of layers $n_L$** | `python script/AE_VAE_experiment_nL.py -dataset_name FreyFace -model_name AE -n_L 3 7 -D_z 5` |
| [`AE_VAE_experiment_Dz.py`](AE_VAE_experiment_Dz.py) | Trains AEs or VAEs with a **fixed number of layers $n_L$** and **several latent dimensions $D_z$** | `python script/AE_VAE_experiment_Dz.py -dataset_name FashionMNIST -model_name VAE -n_L 3 -D_z 2 5` |
| [`beta_VAE_experiment.py`](beta_VAE_experiment.py) | Trains β-VAEs with **fixed $n_L$ and $D_z$** and **several values of β** | `python script/beta_VAE_experiment.py -dataset_name MNIST -model_name VAE -beta 0.02 1 1.5 10 -n_L 2 -D_z 2` |

**Losses:**
- **Training:** AEs are trained with the **BCE** reconstruction loss, and VAEs by maximising the **ELBO** (reconstruction + KL divergence, weighted by β for β-VAEs).
- **Evaluation:** both models are evaluated with **BCE**, **MSE** and **L1** losses.

## 2. Analysis scripts

Once the models are trained, these two scripts produce all the figures in one command:

| Script | Models | Example |
|---|---|---|
| [`Run_code_AE_VAE.py`](Run_code_AE_VAE.py) | AE / VAE | `python script/Run_code_AE_VAE.py -dataset_name FreyFace -model_name AE -n_L 1 -D_z 2` |
| [`Run_code_beta_VAE.py`](Run_code_beta_VAE.py) | β-VAE | `python script/Run_code_beta_VAE.py -dataset_name FreyFace -model_name VAE -beta 0.02 1 1.5 10 -n_L 1 -D_z 5` |

They call the following helper modules:

| AE / VAE | β-VAE | Output |
|---|---|---|
| [`parameter.py`](parameter.py) | [`beta_parameter.py`](beta_parameter.py) | Loads the saved models and prints their hyperparameters and number of parameters |
| [`plt_metrics.py`](plt_metrics.py) | [`beta_VAE_metrics.py`](beta_VAE_metrics.py) | Training and validation curves (BCE, MSE, L1, ELBO) |
| [`reconstruction_image.py`](reconstruction_image.py) | [`beta_VAE_reconstruction_image.py`](beta_VAE_reconstruction_image.py) | Original vs reconstructed images |
| [`visualization_latent_space_generation.py`](visualization_latent_space_generation.py) | [`beta_VAE_visualization_latent_space_generation.py`](beta_VAE_visualization_latent_space_generation.py) | Latent-space plots and newly generated images |

## 3. Outputs

Results are written to:

```
results/
├── AE_experiment_<dataset>/AE_L<n_L>/Dz_<D_z>/
├── VAE_experiment_<dataset>/VAE_L<n_L>/Dz_<D_z>/
└── beta-VAE_experiment_<dataset>/beta_<β>/nL_<n_L>/
```

Example results are included for [AE on FashionMNIST](../results/AE_experiment_FashionMNIST), [VAE on FreyFace](../results/VAE_experiment_FreyFace) and [β-VAE on MNIST](../results/beta-VAE_experiment_MNIST).
