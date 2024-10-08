# AE and VAE experiments 

This directory contains Regis Djaha's experiments on AutoEncoders and Variational Autoencoders, for different number of layers $L$ and different latent space sizes $D_z$.


All experiments are run with the [MNIST, FashioMNIST and FreyFace dataset](https://github.com/RegisKonan/ae_vae_understanding/blob/be148a3712598c35b293b91b3a84487c33f84302/src/main/datasets.py), 

### Fixed the value of Dz and varied L

- [AE_VAE_experiment_Dz.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/AE_VAE_experiment_Dz.py): this is a script that runs AEs and VAEs with different number of layer encoder-decoder architectures

### Fixed the value of L and varied Dz

- [AE_VAE_experiment_nL.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/AE_VAE_experiment_nL.py): this is a script that runs AEs and VAEs with different number of dimension  encoder-decoder architectures


######### These codes allow us to be more productive.

## Descriptions

For each experiment, using the [reconstruction_image.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/reconstruction_image.py), [visualization_latent_space&generation.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/visualization_latent_space%26generation.py),[plt_metrics.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/plt_metrics.py) scripts, we obtain image reconstructions, loss functions, latent spaces, and generated new data. It is worth noting that, in general, we used BCE and MSE metrics to train the AE model and ELBO to train the VAE model. However, for these experiments  in particular, we used the BCE metric to train the AE model and ELBO for the VAE model. For evaluation, we used BCE, MSE, and L1 loss for both cases.

These experiments are contained in the directories [AE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/0f0cdf76304aff5f150e4252253f55d3a820ffe2/results/AE_experiment_FashionMNIST) and [VAE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/0f0cdf76304aff5f150e4252253f55d3a820ffe2/results/VAE_experiment_FreyFace). 


# Beta VAE experiments 

- [beta-VAE_experiment.py](https://github.com/iurteaga/vae_understanding/blob/8bc9ec6a21b75a0da092fa43aef4e499004ee787/script/beta_VAE_experiment.py): this is a script that runs  $\beta$-VAEs with different $\beta$ values.

#####This code allows us to vary different desired values of beta.

### Descriptions

With the help of these codes, [beta_VAE_reconstruction_images.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/beta_VAE_reconstruction_image.py),[beta_VAE_visualization_latent_space_generation.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/beta_VAE_visualization_latent_space_generation.py),[beta_VAE_metrics.py](https://github.com/RegisKonan/ae_vae_understanding/blob/0627aae638a57a1a9a43188e3d7c1fa1a6963978/script/beta_VAE_metrics.py) 
which fix L and Dz, we can simultaneously obtain all the information regarding the various values of beta.


These experiments are contained in the directories [beta-VAE_experiment](https://github.com/RegisKonan/ae_vae_understanding/tree/0f0cdf76304aff5f150e4252253f55d3a820ffe2/results/beta-VAE_experiment_MNIST)


