import pickle
import os
import sys
import matplotlib.pyplot as plt

# My source import
sys.path.append('./src')
from utilities.plotting import VAE_plot_mean_std, VAE_plot_results, VAE_print_evaluation_table



def run_plot_beta_metrics(dataset_info,model,metrics=None):


    # Define the base directory
    
    base_directory  = f'./results/beta-{model["model_name"]}_experiment_{dataset_info['name']}/'
    


    base_directory  = base_directory.format(dataset_name=dataset_info['name'])
    
    min_indices = []
    min_values = []
    
    for beta in model['beta_values']:
        beta_directory = os.path.join(base_directory, f'beta_{beta}')
        nl_directory = os.path.join(beta_directory, f'nL_{model["n_L"]}')
        #file_path = os.path.join(directory , f'{model['model_name']}_fold_results_L{model['n_L']}_dz{model['D_z']}_beta{beta}.pkl')
        
        # Create directories if they do not exist
        os.makedirs(nl_directory, exist_ok=True)
        file_path = os.path.join(beta_directory , f'{model['model_name']}_fold_results_L{model['n_L']}_dz{model['D_z']}_beta{beta}.pkl')
        result=_run_single_experiment(file_path,nl_directory, dataset_info,model, beta,metrics)
        if result is not None:
            min_index, min_value = result
            min_indices.append(min_index)
            min_values.append(min_value)
    return min_indices, min_values


def _run_single_experiment(file_path,directory, dataset_info, model, beta,metrics):
    try:
        with open(file_path, 'rb') as f:
            fold_results = pickle.load(f)

        for loss_type in metrics :
            result_data = VAE_plot_results(fold_results, loss_type)

            fig, ax = plt.subplots(figsize=(12, 8))
            VAE_plot_mean_std(result_data['train_losses'], 'Training and Validation Loss', ax, 'Train VAE Loss', result_data['epochs'])
            VAE_plot_mean_std(result_data['val_losses'], 'Training and Validation Loss', ax, 'Valid VAE Loss', result_data['epochs'])
            VAE_plot_mean_std(result_data['train_recons_losses'], 'Training and Validation Recontruction Loss', ax, 'Train Recons loss', result_data['epochs'])
            VAE_plot_mean_std(result_data['val_recons_losses'], 'Training and Validation Recontruction Loss', ax, 'Validation Recons Loss', result_data['epochs'])
            VAE_plot_mean_std(result_data['train_kld_losses'], 'Training and Validation KLD Loss', ax, 'Train kld loss', result_data['epochs'])
            VAE_plot_mean_std(result_data['val_kld_losses'], 'Training and Validation KLD Loss', ax, 'Validation kld Loss', result_data['epochs'])
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.set_title(f'Training and Validation Loss with Standard Deviation ({loss_type})')
            ax.legend()

            # Save the plot 
            plot_file_path = os.path.join(directory, f'{dataset_info['name']}_{loss_type}_metric_L{model['n_L']}_dz{model['D_z']}_beta{beta}.pdf')
            plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
            plt.show()

            min_index, min_value=VAE_print_evaluation_table(
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

    return min_index, min_value