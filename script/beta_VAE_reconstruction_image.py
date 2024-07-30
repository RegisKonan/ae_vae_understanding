import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
import os

#import pickle

import pickle

sys.path.append('./src')

from utilities.trained_models import  visualize_original_images, VAE_visualize_reconstructions


def run_beta_reconstruction(dataset_info,model,img_shape):


    # Define the base directory
    base_directory  = f'./results/beta-{model["model_name"]}_experiment_{dataset_info['name']}/'


    base_directory  = base_directory.format(dataset_name=dataset_info['name'])
    
    
    for beta in model['beta_values']:
            beta_directory = os.path.join(base_directory, f'beta_{beta}')
            nl_directory = os.path.join(beta_directory, f'nL_{model["n_L"]}')

            
            # Create directories if they do not exist
            os.makedirs(nl_directory, exist_ok=True)
            
            file_path = os.path.join(beta_directory, f'{model["model_name"]}_models_L{model["n_L"]}_dz{model["D_z"]}_beta{beta}.pkl')
            _run_single_experiment(dataset_info,img_shape,file_path, nl_directory)
            
       
def _run_single_experiment(dataset_info,img_shape,file_path, directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = dataset_info['data'].test_loader

    try:
        with open(file_path, 'rb') as f:
            models = pickle.load(f)
            
        visualize_original_images(test_loader, img_shape,directory=directory)
        
        for fold_index, model_to_visualize in enumerate(models):
                print(f"Visualizing reconstructions for fold {fold_index + 1}")
                VAE_visualize_reconstructions(
                                              model_to_visualize,
                                              test_loader,
                                              img_shape, 
                                              L=1,  ### number of sample
                                              device=device,
                                              directory=directory, 
                                              num_samples=10, 
                                              fold_index=fold_index, 
                                              figure_size=(8, 2)
                                              )

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print("An error occurred:", e)
   
    return 


