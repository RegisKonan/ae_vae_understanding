# General imports
import torch
import torch.nn as nn
import sys
import os
import pickle
import argparse
#import pdb

# My source imports
sys.path.append('./src')
from main.loss_function import AE_criterion, VAE_criterion
from utilities.trained_models import AE_k_fold_cross_validation, VAE_k_fold_cross_validation
from models import VAE, AE
from main.datasets import MNISTLoader, FashionMNISTLoader, FreyFaceLoader




def run_experiment(dataset_info, model, optimization, gaussian_distribution,base_directory):
    for n_L in model['L_values']:
        directory = os.path.join(base_directory, f'{model['model_name']}_L{n_L}')
        _run_single_experiment(dataset_info, model, n_L, optimization,gaussian_distribution,directory)

def _run_single_experiment(dataset_info, model, n_L, optimization,gaussian_distribution,directory):
    seed = optimization['seed']
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    train_dataset = dataset_info['data'].train_loader.dataset #train dataset

    test_dataset = dataset_info['data'].test_loader.dataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model['model_name']
    model_config = {'D_x': dataset_info['D_x'], 'n_layers': n_L, 'D_z': model['D_z'], 'activation': model['activation']}
    learning_rate = dataset_info['learning_rate']
    batch_size = dataset_info['batch_size']
    

    
    if model['model_name']=='AE':
           
        training_setup = {
            'epochs': optimization['epochs'],
            'AE_criterion': AE_criterion,
            'device': device,
            'AE_likelihood': dataset_info['AE_likelihood']
        }
        
         
        models, fold_results = AE_k_fold_cross_validation(
                                            model_class=getattr(sys.modules[__name__], model_name),
                                            model_config=model_config,
                                            train_dataset=train_dataset,
                                            test_dataset=test_dataset,
                                            AE_criterion=AE_criterion,
                                            learning_rate=learning_rate,
                                            training_setup=training_setup,
                                            k_folds=optimization['k_folds'],
                                            batch_size=batch_size,
                                            eval_criterions={
                                                'MSE': torch.nn.MSELoss(),
                                                'L1 Loss': torch.nn.L1Loss(),
                                                'BCE': torch.nn.BCELoss()
                                            })

        
        
    elif model['model_name']=='VAE' :
        
        training_setup = {
            'epochs': optimization['epochs'],
            'VAE_criterion': VAE_criterion,
            'device': device,
            'VAE_likelihood': dataset_info['VAE_likelihood']
        }

        if model['kl_div'] == "standard_gaussian":
            
            models, fold_results = VAE_k_fold_cross_validation(
                                                        model_class=getattr(sys.modules[__name__], model_name),
                                                        model_config=model_config,
                                                        train_dataset=train_dataset,
                                                        test_dataset=test_dataset,
                                                        VAE_criterion=VAE_criterion,
                                                        learning_rate=learning_rate,
                                                        training_setup=training_setup,
                                                        beta=1,
                                                        L=1,
                                                        k_folds=optimization['k_folds'],
                                                        batch_size=batch_size,
                                                        kl_div=model['kl_div'],
                                                        mean_2=None,
                                                        var_2=None,
                                                        eval_criterions={
                                                            'MSE': torch.nn.MSELoss(),
                                                            'L1 Loss': torch.nn.L1Loss(),
                                                            'BCE': torch.nn.BCELoss()
                                                        })
        elif model['kl_div'] == "mean_conditional_gaussian":

            models, fold_results = VAE_k_fold_cross_validation(
                                                        model_class=getattr(sys.modules[__name__], model_name),
                                                        model_config=model_config,
                                                        train_dataset=train_dataset,
                                                        test_dataset=test_dataset,
                                                        VAE_criterion=VAE_criterion,
                                                        learning_rate=learning_rate,
                                                        training_setup=training_setup,
                                                        beta=1,
                                                        L=1,
                                                        k_folds=optimization['k_folds'],
                                                        batch_size=batch_size,
                                                        kl_div=model['kl_div'],
                                                        mean_2=gaussian_distribution['mean'],
                                                        var_2=torch.pow(gaussian_distribution['std'], 2),
                                                        eval_criterions={
                                                            'MSE': torch.nn.MSELoss(),
                                                            'L1 Loss': torch.nn.L1Loss(),
                                                            'BCE': torch.nn.BCELoss()
                                                        })  
        else:
            raise ValueError("Invalid KLD loss type. Supported types are 'standard_gaussian' and 'mean_conditional_gaussian'.")
        
    else:
            raise ValueError("Invalid models type. Supported types are 'AE' and 'VAE'.")

    os.makedirs(directory, exist_ok=True)

    model_parameter_path = os.path.join(directory, f'{model['model_name']}_model_parameter_L{n_L}_dz{model['D_z']}.pkl')
    models_file_path = os.path.join(directory, f'{model['model_name']}_models_L{n_L}_dz{model['D_z']}.pkl')
    fold_results_file_path = os.path.join(directory, f'{model['model_name']}_fold_results_L{n_L}_dz{model['D_z']}.pkl')
    

    with open(model_parameter_path, 'wb') as f:
        pickle.dump({
            'device': device,
            'model_name': model_name,
            'model_config': model_config,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'training_setup': training_setup
        }, f)

    with open(models_file_path, 'wb') as f:
        pickle.dump(models, f)

    with open(fold_results_file_path, 'wb') as f:
        pickle.dump(fold_results, f)

if __name__ == '__main__':
    # Example run:
    #   python3 VAE_experiment_Dz.py -dataset_name MNIST
    
    # Input parser
    # TODO: for some unknown reason, the app and flags packages do not like integer arguments
    parser = argparse.ArgumentParser(description='Execute VAE experiment with provided dataset')
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
    
    # If you want, you can use more arguments like this
    
    ## We vary the n_L and set Dz 
    
    parser.add_argument(
        '-n_L',
        type=int,
        nargs='+', # it can take more than one value
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
   # dataset_name = 'FreyFace'  # Change it to 'MNIST' or 'FashionMNIST' or 'FreyFace'
    dataset_name = args.dataset_name
    name = args.model_name
    n_L_values = args.n_L
    D_z= args.D_z
  
    
    batch_size = 100
    learning_rate = 1e-3

    ## Dataset
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
        'AE_likelihood':'bernoulli' if dataset_name != 'FreyFace' else 'gaussian',  ##  Choose 'bernoulli' or 'gaussian' on the data
        'VAE_likelihood':'bernoulli' if dataset_name != 'FreyFace' else 'gaussian',  ##  Choose 'bernoulli' or 'gaussian' on the data
        'D_x': data_d_x
    }

    
     # Model configuration
    model = {
        'model_name': name,  ## Don't change that
        'L_values': n_L_values,# Change it to any list of desired dimensions
        'D_z': D_z ,  # Change this if you want to fix the number of layers to a specific value
        'activation':nn.ReLU,
        'kl_div':  'standard_gaussian' #'standard_gaussian'  # Choose 'standard_gaussian' or 'mean_conditional_gaussian'
    }
    

    #pdb.set_trace()
    # Optimization parameters
    optimization = {
        'k_folds': 10,
        'epochs': 200 if dataset_name != 'FreyFace' else 2000, ### epoch=2000 for FreyFace and 200 for MNIST and FashionMNIST
        'seed' : 7777
    }

    # Gaussian distribution parameters ( Prior VAE)
     
    ### mean_1
    mean_2 =  torch.tensor([5,10], dtype=torch.float32) 
    std_2 = torch.ones((1, 2), dtype=torch.float32) 
    
    
    ### mean_2
    #mean_2 = torch.tensor([[1,1]], dtype=torch.float32)

    #std_2 = torch.ones((1, 2), dtype=torch.float32)
    
    ### std
    # mean_2  = torch.zeros((1, 2), dtype=torch.float32)
    # std_2 = torch.tensor([1,3], dtype=torch.float32)
    
    
    ### mean and std 1
    #mean_2  = torch.tensor([2,5], dtype=torch.float32)
    #std_2 = torch.tensor([3,1], dtype=torch.float32)
    
    
    
    gaussian_distribution = {
        'mean': mean_2,
        'std': std_2
    }
    # Base directory for saving results
    
    results_directory = './results/'
    base_directory = f'{results_directory}/{model["model_name"]}_experiment_{dataset_name}/'
    
    

    os.makedirs(base_directory, exist_ok=True)

    # Run the experiment

    ### If you choose standard_gaussian, make sure that gaussian_distribution = None otherwise gaussian_distribution=gaussian_distribution
    run_experiment(dataset_info,
                       model,
                       optimization=optimization,
                       gaussian_distribution=None, #None # If you choose standard_gaussian, make sure that gaussian_distribution = None otherwise gaussian_distribution=gaussian_distribution
                       base_directory=base_directory)

