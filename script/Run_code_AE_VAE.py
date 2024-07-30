import torch
import sys
import numpy as np

sys.path.append('./src')
from main.datasets import MNISTLoader, FashionMNISTLoader, FreyFaceLoader
sys.path.append('./script')
from plt_metrics import run_plot_metrics
from visualization_latent_space_generation import run_latent_space
from reconstruction_image import run_reconstruction_image
from parameter import run_parameter
import argparse



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Execute AE and VAE experiment with provided dataset')
    # If you want, you can use more arguments like this
    parser.add_argument(
        '-dataset_name',
        type=str,
        choices=['MNIST', 'FashionMNIST', 'FreyFace'],
        required=True,
        help='Dataset name to run experiment with'
    )
    
    parser.add_argument(
        '-model_name',
        type=str,
        choices=['AE', 'VAE'],
        required=True,
        help='Name of the model to use (AE or VAE)'
    )
    
    parser.add_argument(
        '-n_L',
        type=int,
        default=2,
        help='number of layer'
    )
    parser.add_argument(
        '-D_z',
        type=int,
        default=2,
        help='Dimensionality of the latent space'
    )

    # Get arguments
    args = parser.parse_args()
    
        ### Experiment setup
    # dataset_name = 'FreyFace'  # Change it to 'MNIST' or 'FashionMNIST'  or 'FreyFace'
    batch_size = 100
    learning_rate = 1e-3
    dataset_name = args.dataset_name
    name = args.model_name
    D_z= args.D_z
    n_L= args.n_L

    # Load dataset
    if dataset_name == 'MNIST':
        data = MNISTLoader(batch_size=batch_size)
        data_d_x=784
    elif dataset_name == 'FashionMNIST':
        data = FashionMNISTLoader(batch_size=batch_size)
        data_d_x=784
    elif dataset_name == 'FreyFace':
        data = FreyFaceLoader(root='./data/FreyFace', batch_size=batch_size)
        data_d_x=560
    else:
        raise ValueError("Unsupported dataset type: {}".format(dataset_name))


    # Dataset information
    dataset_info = {
        'name': dataset_name,
        'data': data,
        'batch_size': batch_size,
        'learning_rate':learning_rate,
        'D_x': data_d_x
    }

    # Model configuration
    model = {
        'model_name': name,  ## Don't change that
        'n_L': n_L ,  # Modify the number of layer according to the experience carry out in ‘AE’ or ‘VAE’.
        'D_z': D_z,# Modify the dimension according to the experience carry out in ‘AE’ or ‘VAE’.
        'kl_div': 'standard_gaussian' #'standard_gaussian'  # Choose 'standard_gaussian' or 'mean_conditional_gaussian'
    }

    # Gaussian distribution parameters
    
    
    # Also changes the correct values for x1, x2, mean, and std if we use ‘mean_conditional_gaussian’. 
    x1 = np.linspace(-5, 5, 1000) 
    x2=  np.linspace(-5, 5, 1000)
    
    mean  = torch.tensor([0,0], dtype=torch.float32)
    std = torch.tensor([5,1], dtype=torch.float32)

    

    gaussian_distribution = {
        'mean': mean,
        'std': std,
        'x1': x1,
        'x2':x2,
    }



    ############## Latent Space ###################################### 

    min_index, min_value = run_plot_metrics(dataset_info,
                                            model, 
                                            metrics=['BCE'] ### [BCE] for Bernoulli or [MSE] for Gaussian, depending on what you use to train the model.
                                            ) 
    best_model_index = min_index      ### index of the best model   

    ### If you choose standard_gaussian, make sure that gaussian_distribution = None otherwise gaussian_distribution=gaussian_distribution

    ### Use img1=28,img2=20, supervised=False if dataset_name = 'FreyFace' Otherwise  img1=28,img2=28, supervised=True 

    run_latent_space(dataset_info,
                    model, 
                    best_model_index,
                    gaussian_distribution=None, #gaussian_distribution, #None, ##  If you choose standard_gaussian, make sure that gaussian_distribution = None otherwise gaussian_distribution=gaussian_distribution
                    img1=28, # img1=28 for MNIST',  FashionMNIST' and FreyFace dataset
                    img2=28 if dataset_name != 'FreyFace' else 20, # img2=28 for MNIST' and FashionMNIST' dataset and img2=20 for FreyFace dataset
                    supervised=True if dataset_name != 'FreyFace' else False # supervised=True  for MNIST' and FashionMNIST' dataset and supervised=False for FreyFace dataset
                    )



    ############## plot metrics and ELBO ###################################### 

    metrics = ['BCE', 'MSE', 'L1 Loss'] ## Plot the various metrics= [‘BCE’, ‘MSE’, ‘L1 Loss’] you want.


    min_index, min_value = run_plot_metrics(dataset_info, model, metrics)
    print("Returned Index of the minimum validation loss:", min_index)
    print("Returned Value of the minimum validation loss:", min_value)



    ##############  Visualize Original and reconstructions images  ###################################### 



    images= run_reconstruction_image(dataset_info,
                                    model,
                                    img_shape=(28, 28) if dataset_name != 'FreyFace' else (28, 20) ### Use (28,28) for MNIST',  FashionMNIST' and  Otherwise (28,20) for FreyFace dataset
                                    ) 


    ############## parameters and hyperparameters  ###################################### 

    parameter = run_parameter(dataset_info,model)



    