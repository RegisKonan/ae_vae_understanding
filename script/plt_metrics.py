import pickle
import torch
import os
import sys
import matplotlib.pyplot as plt

# My source import
sys.path.append('./src')
from utilities.plotting import AE_plot_mean_std, AE_plot_results, AE_print_evaluation_table, VAE_plot_mean_std, VAE_plot_results, VAE_print_evaluation_table



    
def run_plot_metrics(dataset_info, model, metrics):
    

    
    base_directory = f'./results/{model["model_name"]}_experiment_{dataset_info["name"]}/{model["model_name"]}_L{model["n_L"]}'
    base_directory  = base_directory.format(dataset_name=dataset_info['name'])
    directory = os.path.join(base_directory, f'Dz_{model["D_z"]}')
    
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    

    file_path = os.path.join(base_directory, f'{model['model_name']}_fold_results_L{model['n_L']}_dz{model['D_z']}.pkl') 
   
    min_index = None
    min_value = None

    ##### BCE Metric ######

    if model['model_name'] == 'AE' :
 

        # Load the models from the pickle file
        try:
            with open(file_path, 'rb') as f:
                fold_results = pickle.load(f)


        ##### BCE Metric ######
 
            for loss_type in metrics: # We can change this to our desired loss type
                result_data = AE_plot_results(fold_results, loss_type)

                fig, ax = plt.subplots(figsize=(12, 8))
                AE_plot_mean_std(result_data['train_losses'], 'Training and Validation Loss', ax, 'Train', result_data['epochs'])
                AE_plot_mean_std(result_data['val_losses'], 'Training and Validation Loss', ax, 'Validation', result_data['epochs'])
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training and Validation Loss with Standard Deviation ({loss_type})')
                ax.legend()
                
                # Save the plot 
                plot_file_path = os.path.join(directory, f'{model['model_name']}_{loss_type}_metric_L{model['n_L']}_dz{model['D_z']}.pdf')
                
                plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
                plt.show()
                
                
                ### Print table to see the best model
                min_index,min_value=AE_print_evaluation_table(result_data['valid_result'], result_data['test_result'])


        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)


        ##### L1 Loss Metric ######

    elif model['model_name'] == 'VAE' :

        try:
            with open(file_path, 'rb') as f:
                fold_results = pickle.load(f)

            for loss_type in metrics : # We can change this to our desired loss type
                result_data = VAE_plot_results(fold_results, loss_type)

                fig, ax = plt.subplots(figsize=(12, 8))
                VAE_plot_mean_std(result_data['train_losses'], 'Training and Validation Loss', ax, 'Train VAE Loss', result_data['epochs'])
                VAE_plot_mean_std(result_data['val_losses'], 'Training and Validation Loss', ax, 'Valid VAE Loss', result_data['epochs'])
                VAE_plot_mean_std(result_data['train_recons_losses'], 'Training and Validation Reconstruction Loss', ax, 'Train Recons loss', result_data['epochs'])
                VAE_plot_mean_std(result_data['val_recons_losses'], 'Training and Validation Reconstruction Loss', ax, 'Validation Recons Loss', result_data['epochs'])
                VAE_plot_mean_std(result_data['train_kld_losses'], 'Training and Validation KLD Loss', ax, 'Train kld loss', result_data['epochs'])
                VAE_plot_mean_std(result_data['val_kld_losses'], 'Training and Validation KLD Loss', ax, 'Validation kld Loss', result_data['epochs'])
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.set_title(f'Training and Validation Loss with Standard Deviation ({loss_type})')
                ax.legend()

                # Save the plot 
                plot_file_path = os.path.join(directory, f'{model['model_name']}_{loss_type}_metric_L{model['n_L']}_dz{model['D_z']}.pdf')
                plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
                plt.show()
                
                ### Print table to see the best model
                min_index, min_value =VAE_print_evaluation_table(
                    result_data['valid_metrics'],
                    result_data['test_metrics'],
                    result_data['valid_result'],
                    result_data['test_result'],
                    result_data['recons_valid_result'],
                    result_data['recons_test_result'],
                    result_data['kld_valid_result'],
                    result_data['kld_test_result']
                )
                
 

        except FileNotFoundError:
            print(f"File '{file_path}' not found.")
        except Exception as e:
            print("An error occurred:", e)
        
    
    else:
        raise ValueError("Invalid models type. Supported types are 'AE' and 'VAE'.")
            
            
    return min_index, min_value


