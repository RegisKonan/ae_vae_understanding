import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pickle
import sys

sys.path.append('./src')

from models import AE, VAE
from utilities.latent_space_sample import latent_space_VAE_model, generate_data_from_VAE_latent_space,visualize_latent_space


from utilities.latent_space_sample import latent_space_VAE_model, generate_data_from_VAE_latent_space,visualize_latent_space




def run_latent_space(dataset_info, model, best_model_indices, gaussian_distribution,img1,img2,supervised):
       
    # Define the base directory
    base_directory = f'./results/beta-{model["model_name"]}_experiment_{dataset_info["name"]}/'
    
    base_directory  = base_directory.format(dataset_name=dataset_info['name'])
    
    for beta in model['beta_values']:
        index = best_model_indices[model['beta_values'].index(beta)]
        beta_directory = os.path.join(base_directory, f'beta_{beta}')
        nl_directory = os.path.join(beta_directory, f'nL_{model["n_L"]}')
        
        # Create directories if they do not exist
        os.makedirs(nl_directory, exist_ok=True)
        
        file_path = os.path.join(beta_directory, f'{model["model_name"]}_models_L{model["n_L"]}_dz{model["D_z"]}_beta{beta}.pkl')
        _run_single_experiment(dataset_info, model, index, beta, gaussian_distribution, supervised, img1, img2, file_path, nl_directory)
    
def _run_single_experiment(dataset_info, model, index,  beta, gaussian_distribution,supervised,img1,img2, file_path, directory):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = dataset_info['data'].test_loader
    
    ### beta VAE model in 2 dimensions latent space
    
    if model['D_z'] ==2:

        try:
            with open(file_path, 'rb') as f:
                models = pickle.load(f)
                
            best_model = models[index]


            best_latent_figure = latent_space_VAE_model(dataset_info['name'],
                                                        best_model, 
                                                        test_loader, 
                                                        device, 
                                                        number_fold=index+1,
                                                        kl_div=model['kl_div'],
                                                        gaussian_distribution=gaussian_distribution,
                                                        supervised=supervised
                                                        )
  
            
            # Save the plot 
            plot_file_path = os.path.join(directory, f'{model["model_name"]}_best_latent_space_L{model['n_L']}_dz{model["D_z"]}_beta{beta}_fold{index+1}.png')
            best_latent_figure.savefig(plot_file_path, format="png", bbox_inches="tight")
            plt.show()
        except IndexError:
            print(f"Index {index} out of range.")
        except Exception as e:
            print(f"An error occurred for index {index}:", e)
            
        ##### Sample new data######################
        
        #### Random generation of new data

        # Load the models from the pickle file
        try:
            
            
            generated_image = generate_data_from_VAE_latent_space(best_model,
                                                                  gaussian_distribution, 
                                                                  model['D_z'], 
                                                                  image_to_sample=100,
                                                                  nrow=10,
                                                                  img1=img1,
                                                                  img2=img2,
                                                                  kl_div=model['kl_div']
                                                                  )



            # Display the image
            plt.imshow(generated_image)
            plt.axis('off')
                # Define the file path for saving the generated image
            
            generated_image_path = os.path.join(directory, f'{model["model_name"]}_generated_image_L{model['n_L']}_dz{model["D_z"]}_beta{beta}_fold{index+1}.png')

            # Save the generated image
            generated_image.save(generated_image_path)
            plt.show()

        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
        
        ##### Fix a sample of new data beta-VAE ######################
        
        ### Generate new data using the grid
                 
        
        try:

            sample = visualize_latent_space(best_model,
                                            device,
                                            img1=img1,
                                            img2=img2
                                            )


            # Display the image
            plt.imshow(sample)
            plt.axis('off')
                # Define the file path for saving the generated image
            generated_image_path = os.path.join(directory, f'{model["model_name"]}_generated_image_L{model['n_L']}_dz{model["D_z"]}_beta{beta}_fold{index+1}.png')

            # Save the generated image
            sample.save(generated_image_path)
            plt.show()

        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
    
         ## beta-Variational Autoencoder model without latent space for dimensions greater than two.
         
    else : 
            
        try:
            with open(file_path, 'rb') as f:
                models = pickle.load(f)
            
            best_model = models[index]
            generated_image = generate_data_from_VAE_latent_space(best_model,
                                                                    gaussian_distribution, 
                                                                    model['D_z'],
                                                                    image_to_sample=100, 
                                                                    nrow=10,
                                                                    img1=img1, 
                                                                    img2=img2,
                                                                    kl_div=model['kl_div']
                                                                    )
            
            plt.imshow(generated_image)
            plt.axis('off')
            generated_image_path = os.path.join(directory, f'{model["model_name"]}_generated_image_L{model['n_L']}_dz{model["D_z"]}_fold{best_model_index+1}.png')
            plt.savefig(generated_image_path, format="png", bbox_inches="tight")
            plt.show()
            
        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
            

         ### Generate new data using the grid
                 
        
        try:

            sample = visualize_latent_space(best_model,
                                            device,
                                            img1=img1,
                                            img2=img2
                                            )


            # Display the image
            plt.imshow(sample)
            plt.axis('off')
                # Define the file path for saving the generated image
            generated_image_path = os.path.join(directory, f'{model["model_name"]}_generated_image_L{model['n_L']}_dz{model["D_z"]}_beta{beta}_fold{index+1}.png')

            # Save the generated image
            sample.save(generated_image_path)
            plt.show()

        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
    
    return