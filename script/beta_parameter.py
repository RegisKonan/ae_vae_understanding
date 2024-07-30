import pickle
import os

    
    
def run_beta_parameter(dataset_info,model):
    

    # Define the base directory
    base_directory = f'./results/beta-{model['model_name']}_experiment_{dataset_info['name']}/'
    


    base_directory = base_directory.format(dataset_name=dataset_info['name'])

    for beta in model['beta_values']:
        directory = os.path.join(base_directory, f'beta_{beta}')
        models_file_path = os.path.join(directory, f'{model['model_name']}_models_L{model['n_L']}_dz{model['D_z']}_beta{beta}.pkl')
        _run_single_experiment(models_file_path,directory,model, beta)
        
    

def _run_single_experiment(models_file_path,directory, model, beta):


    # Load models from the pickle file
    try:
        with open(models_file_path, 'rb') as f:
            models = pickle.load(f)

        # Count the number of parameters in the first model
        total_params = sum(p.numel() for p in models[0].parameters())
        print(f"Number of parameters in the model: {total_params}")

    except FileNotFoundError:
        print(f"File '{models_file_path}' not found.")
    except Exception as e:
        print("An error occurred:", e)

    # File path for loading model parameters
    model_param_file_path = os.path.join(directory, f'{model['model_name']}_model_parameter_L{model['n_L']}_dz{model['D_z']}_beta{beta}.pkl')

    try:
        # Load data from the pickle file
        with open(model_param_file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        # Access the loaded data
        loaded_device = loaded_data['device']
        model_name = loaded_data[f'{model['model_name']}_model_name']
        model_config = loaded_data[f'{model['model_name']}_model_config']
        learning_rate = loaded_data['learning_rate']
        batch_size = loaded_data['batch_size']
        training_setup = loaded_data['training_setup']

        # Print the loaded parameters
        print('device:', loaded_device)
        print('model_name:', model_name)
        print('model_config:', model_config)
        print('learning_rate:', learning_rate)
        print('batch_size:', batch_size)
        print('training_setup:', training_setup)

    except FileNotFoundError:
        print(f"File '{model_param_file_path}' not found.")
    except Exception as e:
        print("An error occurred:", e)


    return
