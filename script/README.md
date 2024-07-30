# AE and VAE experiments 

This directory contains Regis Djaha's experiments on AutoEncoders and Variational Autoencoders, for different number of layers $L$ and different latent space sizes $D_z$.


All experiments are run with the [MNIST dataset](https://github.com/iurteaga/vae_understanding/blob/e22cb4949e03860d98fe1be43536ed3995ea6078/src/main/MNIST_dataset.py), [FashioMNIST_dataset](https://github.com/iurteaga/vae_understanding/blob/e22cb4949e03860d98fe1be43536ed3995ea6078/src/main/FashionMNIST_dataset.py) and [FreyFace_dataset]

### Fixed the value of Dz and varied L

- [AE_VAE_experiment_Dz.py](https://github.com/iurteaga/vae_understanding/blob/11a04efdd3f09729cc3f41ed623b86923dafca0e/script/AE_VAE_experiment_Dz.py): this is a script that runs AEs and VAEs with different number of layer encoder-decoder architectures

### Fixed the value of L and varied Dz

- [AE_VAE_experiment_nL.py](https://github.com/iurteaga/vae_understanding/blob/11a04efdd3f09729cc3f41ed623b86923dafca0e/script/AE_VAE_experiment_nL.py): this is a script that runs AEs and VAEs with different number of dimension  encoder-decoder architectures


######### These codes allow us to be more productive.

## Descriptions

For each experiment, using the [reconstruction_image.py](https://github.com/iurteaga/vae_understanding/blob/4a10f0e86df7589f574faf954cbfaee5e6559db9/script/experiment_files/reconstruction_image.py), [visualization_latent_space&generation.py](https://github.com/iurteaga/vae_understanding/blob/6497afc21d95e215bc118ecabf39327aa383ac48/script/experiment_files/visualization_latent_space%26generation.py),[plt_metrics.py](https://github.com/iurteaga/vae_understanding/blob/86df0b577ad41b395182e68d2cdc3d4f4092f847/script/experiment_files/plt_metrics.py) scripts, we obtain image reconstructions, loss functions, latent spaces, and generated new data. It is worth noting that, in general, we used BCE and MSE metrics to train the AE model and ELBO to train the VAE model. However, for these experiments  in particular, we used the BCE metric to train the AE model and ELBO for the VAE model. For evaluation, we used BCE, MSE, and L1 loss for both cases.

These experiments are contained in the directories [AE_experiment](https://github.com/iurteaga/vae_understanding/tree/9f68b944c764780170ddf6b148614e8ba33348a3/script/experiment_files/AE_experiment) and [VAE_experiment](https://github.com/iurteaga/vae_understanding/tree/9f68b944c764780170ddf6b148614e8ba33348a3/script/experiment_files/VAE_experiment). 


# Beta VAE experiments 

- [beta-VAE_experiment1.py](https://github.com/iurteaga/vae_understanding/blob/28a8867baf79768dcf1c084460db9ada6510fc2b/script/experiment_files/beta_VAE_experiment1.py): this is a script that runs AEs with $L=2$ encoder-decoder architectures, $D_z=2$ and $beta=[0.02, 1, 1.5 10]$

#####This code allows us to vary different desired values of beta.

### Descriptions

With the help of these codes, [beta-VAE_reconstruction_images.py](https://github.com/iurteaga/vae_understanding/blob/b63acb1c6862be42c4a45a6e920cfb5e66c0cf76/script/experiment_files/beta_VAE_reconstruction_image.py),[beta-VAE_visualization_latent_space&generation.py](https://github.com/iurteaga/vae_understanding/blob/0d4c7fc17f549e92adbe8039a16b5247ce793757/script/experiment_files/beta-VAE_visualization_latent_space%26generation.py),[beta-VAE_plt_metrics.py](https://github.com/iurteaga/vae_understanding/blob/b00862f4ef44675b48d0b15d4faad8e83783cfc2/script/experiment_files/beta_VAE_metrics.py) 
which fix L and Dz, we can simultaneously obtain all the information regarding the various values of beta.


These experiments are contained in the directories [beta-VAE_experiment](https://github.com/iurteaga/vae_understanding/tree/2557932c050d43197f620633eaf70359ee5625ec/script/experiment_files/beta-VAE_experiment)


## Observed issue

- The issue of this code [VAE_experiment2.py](https://github.com/iurteaga/vae_understanding/blob/97f4953f38d019d12322e5daaab986444af6e942/script/experiment_files/VAE_experiment2.py) is that we cannot fix different values of D_z at the same time because we manually work with the mean and standard deviations.

- The issue of thes codes [reconstruction_image.py](https://github.com/iurteaga/vae_understanding/blob/4a10f0e86df7589f574faf954cbfaee5e6559db9/script/experiment_files/reconstruction_image.py) and [beta-VAE_reconstruction_images.py](https://github.com/iurteaga/vae_understanding/blob/b63acb1c6862be42c4a45a6e920cfb5e66c0cf76/script/experiment_files/beta_VAE_reconstruction_image.py) is that when I try to save the images and their reconstructions, I get blank pages.
