import matplotlib.pyplot as plt
import torch
import os
import pickle

from models import AE, VAE
from utilities.latent_space_sample import latent_space_AE_model, generate_data_from_AE_latent_space, latent_space_VAE_model, generate_data_from_VAE_latent_space, visualize_latent_space

def run_latent_space(dataset_info, model, best_model_index, gaussian_distribution, img1, img2, supervised):
    
    ## test dataset 
    test_loader = dataset_info['data'].test_loader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the base directory
    
    base_directory = f'./results/{model["model_name"]}_experiment_{dataset_info["name"]}/{model["model_name"]}_L{model["n_L"]}'
    base_directory  = base_directory.format(dataset_name=dataset_info['name'])

    directory = os.path.join(base_directory, f'Dz_{model["D_z"]}')
    
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(base_directory , f'{model["model_name"]}_models_L{model["n_L"]}_dz{model["D_z"]}.pkl')
    
    
    ## Autoencoder model
    
    #### Visualize latent space in 2 dimensions
    
    if model['D_z'] == 2:
        if model['model_name'] == 'AE':
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                    
                fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
                fig.suptitle('Latent Space Visualization for each fold', fontsize=16)
                axes = axes.flatten()
                
                for i, ae_model in enumerate(models):
                    latent_space_AE_model(dataset_info['name'], 
                                          ae_model, 
                                          test_loader,
                                          device, 
                                          supervised, 
                                          ax=axes[i]
                                          )
                    
                plot_file_path = os.path.join(directory, f'AE_latent_space_L{model["n_L"]}_dz{model["D_z"]}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
            #### Visualize the best latent space of AE in 2 dimensions
                
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                    
                best_model = models[best_model_index]
                latent_space_AE_model(dataset_info['name'], 
                                      best_model, 
                                      test_loader,
                                      device,
                                      supervised
                                      )
                
                plot_file_path = os.path.join(directory, f'AE_best_latent_space_L{model["n_L"]}_dz{model["D_z"]}_fold{best_model_index+1}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
                #### Random generation of new data
                
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                
                best_model = models[best_model_index]
                generated_image = generate_data_from_AE_latent_space(best_model, D_z=model['D_z'], 
                                                                     img1=img1,
                                                                     img2=img2, 
                                                                     image_to_sample=100,
                                                                     nrow=10)
                
                plt.imshow(generated_image)
                plt.axis('off')
                plot_file_path = os.path.join(directory, f'AE_generated_image_L{model["n_L"]}_dz{model["D_z"]}_fold{best_model_index+1}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
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
                
                plt.imshow(sample)
                plt.axis('off')
                plot_file_path = os.path.join(directory, f'AE_sample_fix_L{model["n_L"]}_dz{model["D_z"]}_fold{best_model_index+1}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
        ## Variational Autoencoder model
    
         #### Visualize the best latent space of VAE in 2 dimensions
                
        elif model['model_name'] == 'VAE':
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                
                best_model = models[best_model_index]
                latent_space_VAE_model(dataset_info['name'],
                                       best_model,
                                       test_loader,
                                       device, 
                                       number_fold=best_model_index+1,
                                       kl_div=model['kl_div'],
                                       gaussian_distribution=gaussian_distribution,
                                       supervised=supervised
                                       )
                
                plot_file_path = os.path.join(directory, f'{model["model_name"]}_best_latent_space_L{model["n_L"]}_dz{model["D_z"]}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
            #### Random generation of new data
                
            try:
                generated_image = generate_data_from_VAE_latent_space(best_model, 
                                                                      gaussian_distribution,
                                                                      model['D_z'],
                                                                      image_to_sample=100,
                                                                      nrow=10, img1=img1, 
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
                
                plt.imshow(sample)
                plt.axis('off')
                sample_path = os.path.join(directory, f'{model["model_name"]}_sample_fix_L{model["n_L"]}_dz{model["D_z"]}_fold{best_model_index+1}.png')
                plt.savefig(sample_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
        else:
            raise ValueError("Invalid model type. Supported types are 'AE' and 'VAE'.")
        
    ## Autoencoder model without latent space for dimensions greater than two.
            
    else: 
        if model['model_name'] == 'AE':
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                
                best_model = models[best_model_index]
                generated_image = generate_data_from_AE_latent_space(best_model,
                                                                     D_z=model['D_z'], 
                                                                     img1=img1,
                                                                     img2=img2, 
                                                                     image_to_sample=100, 
                                                                     nrow=10
                                                                     )
                
                plt.imshow(generated_image)
                plt.axis('off')
                plot_file_path = os.path.join(directory, f'AE_generated_image_L{model["n_L"]}_dz{model["D_z"]}_fold{best_model_index+1}.png')
                plt.savefig(plot_file_path, format="png", bbox_inches="tight")
                plt.show()
                
            except FileNotFoundError:
                print(f"File '{file_path}' not found.")
            except Exception as e:
                print("An error occurred:", e)
                
        ## Variational Autoencoder model without latent space for dimensions greater than two.
                
        elif model['model_name'] == 'VAE':
            try:
                with open(file_path, 'rb') as f:
                    models = pickle.load(f)
                
                best_model = models[best_model_index]
                generated_image = generate_data_from_VAE_latent_space(best_model,
                                                                      gaussian_distribution, model['D_z'],
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
                
        else:
            raise ValueError("Invalid model type. Supported types are 'AE' and 'VAE'.")
    
    return
