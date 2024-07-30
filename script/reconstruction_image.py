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

from utilities.trained_models import visualize_original_images, AE_visualize_reconstructions, VAE_visualize_reconstructions

from models import AE, VAE



def run_reconstruction_image(dataset_info, model,img_shape):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    # Define the base directory
    
    base_directory = f'./results/{model["model_name"]}_experiment_{dataset_info["name"]}/{model["model_name"]}_L{model["n_L"]}'
    directory  = base_directory .format(dataset_name=dataset_info['name'])
    directory = os.path.join(base_directory, f'Dz_{model["D_z"]}')
    
    

    test_loader = dataset_info['data'].test_loader
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(base_directory, f'{model['model_name']}_models_L{model["n_L"]}_dz{model["D_z"]}.pkl')
    
    if model['model_name'] == 'AE' :
    
        # Load the models from the pickle file
        try:
            with open(file_path, 'rb') as f:
                models= pickle.load(f)

            # Iterate over the loaded models
            visualize_original_images(test_loader, img_shape, directory=directory)

            for fold_index, model_to_visualize in enumerate(models):
                print(f"Visualizing reconstructions for fold {fold_index + 1}")
                AE_visualize_reconstructions(
                                             model_to_visualize, 
                                             test_loader,
                                             img_shape,
                                             device=device, 
                                             directory=directory,
                                             num_samples=10, 
                                             fold_index=fold_index , 
                                             figure_size=(8, 2)
                                             )


        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
        
    elif model['model_name'] == 'VAE' :

        try:
            with open(file_path, 'rb') as f:
                models_1_1 = pickle.load(f)

            # Iterate over the loaded models
            visualize_original_images(test_loader, img_shape,directory=directory)
            for fold_index, model_to_visualize in enumerate(models_1_1):
                print(f"Visualizing reconstructions for fold {fold_index + 1}")
                VAE_visualize_reconstructions(
                                              model_to_visualize,
                                              test_loader, 
                                              img_shape, 
                                              L=1,  ## number of sample
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
            
    else:
        raise ValueError("Invalid models type. Supported types are 'AE' and 'VAE'.")



