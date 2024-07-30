import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import sys 
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

sys.path.append('./src')

from main.loss_function import VAE_criterion, AE_criterion
from models import AE, VAE
from main.datasets import data_processing

######## AE Trained ######

def train_AE_model(model, train_loader, val_loader, test_loader, AE_criterion, eval_criterions, optimizer, device, epochs, AE_likelihood, fold):
    train_outputs = []
    train_average_losses = []
    val_average_losses = []

    # Iterate over epochs
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for  data in train_loader:
            img = data_processing(data)
            
            x_in = img.view(img.size(0), -1).to(device)
            z, x_out = model(x_in)
            loss = AE_criterion(x_out, x_in, AE_likelihood)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_loader)
        train_average_losses.append(average_train_loss)

        print(f'Train Fold/Epoch: {fold + 1}/{epoch + 1}, Training Loss: {average_train_loss:.6f}')

        # Validation phase
        average_val_loss = eval_AE_model(fold, model, val_loader, eval_criterions, device)
        average_val_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in average_val_loss.items()}
        val_average_losses.append(average_val_losses_1)

        print(f'Validation Fold/Epoch: {fold + 1}/{epoch + 1}, Validation AE Loss: {average_val_losses_1:}')




        # Keep track of training outputs
        train_outputs.append((epoch, x_in, x_out))

    return train_outputs, train_average_losses, val_average_losses



###### AE Test ######

def eval_AE_model(fold, model, test_loader, criterions, device):
    model.eval()
    criterion_names = list(criterions.keys())
    average_eval_loss = {criterion_name: 0.0 for criterion_name in criterion_names}

    for criterion_name, criterion in criterions.items():
        total_ae_loss = 0.0

        for data in test_loader:
            img = data_processing(data)
            
            img = img.to(device)
            x_in = img.view(img.size(0), -1)
            z, x_out = model(x_in)
            loss = criterion(x_out, x_in)
            total_ae_loss += loss.item()

        # Calculate the average loss
        average_loss = total_ae_loss / len(test_loader)
        average_eval_loss[criterion_name] = average_loss

    return average_eval_loss


##### VAE Trained #########

##### train VAE ####
import pdb
def train_VAE_model(model, train_loader,val_loader, test_loader, VAE_criterion,eval_criterions, optimizer, device, epochs, VAE_likelihood, beta, L, fold,kl_div, mean_2=None, var_2=None):
    outputs = []

    average_recons_losses = []  # List to store average reconstruction losses
    average_kld_losses = []     # List to store average KLD losses
    average_vae_losses = []         # List to store average total losses
    average_vae_val_losses = []
    average_recons_val_losses = []
    average_kld_val_losses = []
    average_val_losses = []

    for epoch in range(epochs):
        model.train()
        total_vae_loss = 0.0
        total_recons_loss = 0.0
        total_kld_loss = 0.0
        for data in train_loader:
            # Just get the features, aka the images, and ignore labels
            img = data_processing(data)
                
            x_in = img.view(img.size(0), -1).to(device)
            z_mu, z_logvar, z_samples, x_out  = model(x_in, L)
            recons_loss, kld_loss, vae_loss = VAE_criterion(kl_div, x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta, mean_2, var_2)
            optimizer.zero_grad()
            vae_loss.backward()
            optimizer.step()
            total_recons_loss += recons_loss.item()
            total_kld_loss += kld_loss.item()
            total_vae_loss += vae_loss.item()

        average_recons_loss = total_recons_loss / len(train_loader)
        average_kld_loss = total_kld_loss / len(train_loader)
        average_vae_loss = total_vae_loss / len(train_loader)

        average_recons_losses.append(average_recons_loss)
        average_kld_losses.append(average_kld_loss)
        average_vae_losses.append(average_vae_loss)


        print(f'Train Fold/Epoch: {fold + 1}/{epoch + 1}, Training Loss:{average_vae_loss:.6f}, Training Recons Loss: {average_recons_loss:.6f}, Training KLD Loss: {average_kld_loss:.6f}')


        # Validation phase 

        #average_loss, average_vae_val_loss, average_recons_val_loss, average_kld_val_loss = eval_VAE_model(fold, model, val_loader, eval_criterions, VAE_likelihood, beta, L, device)
        average_loss, average_vae_val_loss, average_recons_val_loss, average_kld_val_loss = eval_VAE_model(fold, model, val_loader, eval_criterions, VAE_likelihood, beta, L,device,kl_div,mean_2, var_2)
        average_loss_0 = {criterion_name_0: loss_value_0 for criterion_name_0, loss_value_0 in average_loss.items()}
        average_vae_val_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in average_vae_val_loss.items()}
        average_recons_val_losses_2 = {criterion_name_2: loss_value_2 for criterion_name_2, loss_value_2 in average_recons_val_loss.items()}
        average_kld_val_losses_3 = {criterion_name_3: loss_value_3 for criterion_name_3, loss_value_3 in average_kld_val_loss.items()}


        average_val_losses.append(average_loss_0)
        average_vae_val_losses.append(average_vae_val_losses_1)
        average_recons_val_losses.append(average_recons_val_losses_2)
        average_kld_val_losses.append(average_kld_val_losses_3)

        print(f'Validation Fold/Epoch: {fold + 1}/{epoch + 1},  Validation Loss: {average_loss_0}, Validation VAE Loss: {average_vae_val_losses_1}, Validation Recons Loss: {average_recons_val_losses_2}, Validation KLD Loss: {average_kld_val_losses_3}')



        outputs.append((epoch, x_in, x_out))


    return outputs, average_vae_losses, average_recons_losses, average_kld_losses, average_val_losses, average_vae_val_losses, average_recons_val_losses, average_kld_val_losses

##### test VAE ####

def eval_VAE_model(fold, model, test_loader, criterions, VAE_likelihood, beta, L, device,kl_div, mean_2=None, var_2=None):
    model.eval()
    criterion_names = list(criterions.keys())

    average_vae_val_loss = {criterion_name: 0.0 for criterion_name in criterion_names}
    average_recons_val_loss = {criterion_name: 0.0 for criterion_name in criterion_names}
    average_kld_val_loss = {criterion_name: 0.0 for criterion_name in criterion_names}

    average_total_loss = {criterion_name: 0.0 for criterion_name in criterion_names}


    for name, criterion in criterions.items():
        total_recons_loss = 0.0
        total_kld_loss = 0.0
        total_vae_loss = 0.0

        total_loss = 0.0

        for data in test_loader:
            img = data_processing(data)
                
            x_in = img.view(img.size(0), -1).to(device)
            z_mu, z_logvar, z_samples, x_out = model(x_in, L)
            x_out_flat = x_out.view(x_in.size()).to(device)

            loss = criterion(x_out_flat , x_in)


            recons_loss, kld_loss, vae_loss = VAE_criterion(kl_div, x_out, x_in, z_mu, z_logvar, L, VAE_likelihood, beta, mean_2, var_2)

            total_recons_loss += recons_loss.item()
            total_kld_loss += kld_loss.item()
            total_vae_loss += vae_loss.item()

            total_loss += loss.item()

        # Calculate the average loss for this criterion
        average_recons_loss = total_recons_loss / len(test_loader)
        average_kld_loss = total_kld_loss / len(test_loader)
        average_vae_loss = total_vae_loss / len(test_loader)

        average_loss = total_loss / len(test_loader)

        # Update the dictionaries with the average values
        average_vae_val_loss[name] = average_vae_loss
        average_recons_val_loss[name] = average_recons_loss
        average_kld_val_loss[name] = average_kld_loss

        average_total_loss[name] = average_loss

    return average_total_loss, average_vae_val_loss, average_recons_val_loss, average_kld_val_loss



######  Visualize original images AE and and VAE #####
            
def visualize_original_images(test_loader, img_shape, directory,num_samples=10, figure_size=(8, 2)):
    with torch.no_grad():
        for batch in test_loader:
            batch_features = data_processing(batch)
            batch_features = batch_features.view(-1, img_shape[0] * img_shape[1])
            break

    plt.figure(figsize=figure_size)
    plt.gray()

    plt.suptitle('Visualizing Original Images', fontsize=12)

    for i in range(num_samples):
        ax = plt.subplot(2, num_samples, i + 1)
        # Check if there is only one image in the batch
        if len(batch_features) == 1:
            plt.imshow(batch_features[0].numpy().reshape(img_shape))
        else:
            plt.imshow(batch_features[i].numpy().reshape(img_shape))
        plt.axis('off')

   
    
    visualize_path = os.path.join(directory, 'origine_image.png')
    plt.savefig(visualize_path, format='png', bbox_inches='tight')
    
    plt.show()
     
######  Visualize reconstructions images AE #####
def AE_visualize_reconstructions(model, test_loader, img_shape,  device, directory, num_samples=10, fold_index=None, figure_size=(8, 2)):
    with torch.no_grad():
         for batch in test_loader:
            batch_features = data_processing(batch)
            batch_features = batch_features.to(device)
            x_in = batch_features.view(batch_features.size(0), -1)
            z, x_out = model(x_in)
            break

    plt.figure(figsize=figure_size)
    plt.gray()

    for i, (item_in, item_out) in enumerate(zip(x_in.cpu().numpy(), x_out.cpu().numpy())):
        if i >= num_samples:
            break
        # Plot original image
        if fold_index == 0:
          plt.subplot(2, num_samples, i + 1)
          item_in = item_in.reshape(img_shape)
          plt.imshow(item_in)
          plt.axis('off')

        # Plot reconstructed image for all folds
        plt.subplot(2, num_samples, num_samples + i + 1)
        item_out = item_out.reshape(img_shape)
        plt.imshow(item_out)
        plt.axis('off')

    
    # Save the plot
    plt.savefig('{}/AE_reconstruction_images_fold{}.png'.format(directory, fold_index + 1), format='png', bbox_inches='tight')
    plt.show()

######  Visualize reconstructions images VAE #####

def VAE_visualize_reconstructions(model, test_loader, img_shape, L, device, directory, num_samples=10, fold_index=None, figure_size=(8, 2)):
    with torch.no_grad():
        for batch in test_loader:
            batch_features = data_processing(batch)    
            batch_features = batch_features.to(device)
            x_in = batch_features.view(batch_features.size(0), -1)
            z_mu, z_logvar, z_samples, x_out = model(x_in, L)
            break

    plt.figure(figsize=figure_size)
    plt.gray()

    for i, (item_in, item_out) in enumerate(zip(x_in.cpu().numpy(), x_out.cpu().numpy())):
        if i >= num_samples:
            break
        # Plot original image
        if fold_index == 0:
          plt.subplot(2, num_samples, i + 1)
          item_in = item_in.reshape(img_shape)
          plt.imshow(item_in)
          plt.axis('off')

        # Plot reconstructed image for all folds
        plt.subplot(2, num_samples, num_samples + i + 1)
        item_out = item_out.reshape(img_shape)
        plt.imshow(item_out)
        plt.axis('off')
    
    # Save the plot
    plt.savefig('{}/VAE_reconstruction_images_fold{}.pdf'.format(directory, fold_index + 1), format='pdf', bbox_inches='tight')
    plt.show()
    



##### Cross Validation AE ######

def AE_k_fold_cross_validation(model_class, model_config, train_dataset, test_dataset, AE_criterion, learning_rate, training_setup, k_folds, batch_size,
                            eval_criterions={
        'MSE': torch.nn.MSELoss(),
        'L1 Loss': torch.nn.L1Loss(),
        'BCE': torch.nn.BCELoss()
    }):
    # Set up k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize a list to store models and losses
    models = []
    fold_results = []

    # Perform k-fold cross-validation
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into training and validation sets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        # Create data loaders
        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        print(len(train_loader_fold.dataset))
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        print(len(val_loader_fold.dataset))
        test_loader_fold = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(test_loader_fold.dataset)

        # Initialize autoencoder model

        model = model_class(**model_config).to(training_setup['device'])

        # Define loss function and optimizer after initializing the model
        criterion = AE_criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize test_losses_fold
        test_losses_fold_1 = []
        val_losses_fold_1 = []

        # Train the model for k-fold cross-validation
        train_outputs_fold, train_losses_fold, val_losses_fold = train_AE_model(
            model,
            train_loader_fold,
            val_loader_fold,
            test_loader_fold,
            criterion,
            eval_criterions,
            optimizer,
            training_setup['device'],
            training_setup['epochs'],
            training_setup['AE_likelihood'],
            fold
        )

        # Test and validate the model for k-fold cross-validation
        val_losses_fold_1 = eval_AE_model(fold,
                                       model,
                                       val_loader_fold,
                                       eval_criterions,
                                       training_setup['device']
                                       )
        val_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in val_losses_fold_1.items()}
        print(f'Validation results for fold {fold +1}, Validation Average AE Loss: {val_losses_1:}')

        test_losses_fold_1 = eval_AE_model(fold,
                                           model,
                                           test_loader_fold,
                                           eval_criterions,
                                           training_setup['device']
                                           )
        #print(f"Test results for fold {fold + 1}: {test_losses_fold:.6f}")
        test_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in test_losses_fold_1.items()}
        print(f'Test results for fold {fold +1}, Testing Average AE Loss: {test_losses_1}')


        # Save the model from this fold
        models.append(model)
        fold_results.append({
            "outputs": train_outputs_fold,
            "train_losses_fold": train_losses_fold,
            "val_losses_fold": val_losses_fold,
            "val_results": val_losses_1 ,
            "test_results": test_losses_1
        })

    return models, fold_results


###### Cross Validation VAE ###########

def VAE_k_fold_cross_validation(model_class, model_config, train_dataset, test_dataset, VAE_criterion, learning_rate, training_setup, beta, L, k_folds, batch_size,kl_div, mean_2=None, var_2=None,
                            eval_criterions={
        'MSE': torch.nn.MSELoss(),
        'L1 Loss': torch.nn.L1Loss(),
        'BCE': torch.nn.BCELoss()
    }):
    # Set up k-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize a list to store models and losses
    models = []
    fold_results = []

    # Perform k-fold cross-validation
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_dataset)):
        print(f"Fold {fold + 1}/{k_folds}")

        # Split dataset into training and validation sets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)

        # Create data loaders
        train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        print(len(train_loader_fold.dataset))
        val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        print(len(val_loader_fold.dataset))
        test_loader_fold = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(test_loader_fold.dataset)

        # Initialize autoencoder model

        model = model_class(**model_config).to(training_setup['device'])

        # Define loss function and optimizer after initializing the model
        criterion = VAE_criterion
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Initialize test_losses_fold
        test_losses_fold_1 = []
        test_vae_losses_fold_1 = []
        test_recons_losses_fold_1 = []
        test_kld_losses_fold_1 = []
        val_losses_fold_1 = []
        val_vae_losses_fold_1 = []
        val_recons_losses_fold_1 = []
        val_kld_losses_fold_1 = []

        # Train the model for k-fold cross-validation


            
        train_outputs_fold, train_vae_losses_fold, train_recons_losses_fold, train_kld_losses_fold, val_losses_fold, val_vae_losses_fold, val_recons_losses_fold, val_kld_losses_fold = train_VAE_model(
                model,
                train_loader_fold,
                val_loader_fold,
                test_loader_fold,
                criterion,
                eval_criterions,
                optimizer,
                training_setup['device'],
                training_setup['epochs'],
                training_setup['VAE_likelihood'],
                beta,
                L,
                fold,
                kl_div,
                mean_2,
                var_2
            )
        # Test and validate the model for k-fold cross-validation

            
        val_losses_fold_1, val_vae_losses_fold_1, val_recons_losses_fold_1, val_kld_losses_fold_1 = eval_VAE_model(fold,
                                model,
                                val_loader_fold,
                                eval_criterions,
                                training_setup['VAE_likelihood'],
                                beta,
                                L,
                                training_setup['device'],
                                kl_div,
                                mean_2,
                                var_2
                                )
            
        val_losses_0 = {criterion_name_0: loss_value_0 for criterion_name_0, loss_value_0 in val_losses_fold_1.items()}
        val_vae_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in val_vae_losses_fold_1.items()}
        recons_val_losses_2 = {criterion_name_2: loss_value_2 for criterion_name_2, loss_value_2 in val_recons_losses_fold_1.items()}
        kld_val_losses_3 = {criterion_name_3: loss_value_3 for criterion_name_3, loss_value_3 in val_kld_losses_fold_1.items()}
        print(f'Validation results for fold {fold +1}, Validation Loss: {val_losses_0}, Validation Average VAE Loss: {val_vae_losses_1}, Validation Average Recons Loss: {recons_val_losses_2}, Validation Average KLD Loss: {kld_val_losses_3}')



        test_losses_fold_1, test_vae_losses_fold_1, test_recons_losses_fold_1, test_kld_losses_fold_1 = eval_VAE_model(fold,
                                                                        model,
                                                                        test_loader_fold,
                                                                        eval_criterions,
                                                                        training_setup['VAE_likelihood'],
                                                                        beta,
                                                                        L,
                                                                        training_setup['device'],
                                                                        kl_div,
                                                                        mean_2,
                                                                        var_2
                                                                        )
            
        test_losses_0 = {criterion_name_0: loss_value_0 for criterion_name_0, loss_value_0 in test_losses_fold_1.items()}
        test_vae_losses_1 = {criterion_name_1: loss_value_1 for criterion_name_1, loss_value_1 in test_vae_losses_fold_1.items()}
        recons_test_losses_2 = {criterion_name_2: loss_value_2 for criterion_name_2, loss_value_2 in test_recons_losses_fold_1.items()}
        kld_test_losses_3 = {criterion_name_3: loss_value_3 for criterion_name_3, loss_value_3 in test_kld_losses_fold_1.items()}
        print(f'Test results for fold {fold +1}, Testing Loss: {test_losses_0}, Testing Average VAE Loss: {test_vae_losses_1}, Testing Average Recons Loss: {recons_test_losses_2}, Testing Average KLD Loss: {kld_test_losses_3}')


        # Save the model from this fold
        models.append(model)
        fold_results.append({
            "outputs": train_outputs_fold,
            "train_recons_losses_fold": train_recons_losses_fold,
            "train_kld_losses_fold": train_kld_losses_fold,
            "train_losses_fold": train_vae_losses_fold,
            "val_fold": val_losses_fold,
            "val_losses_fold": val_vae_losses_fold,
            "val_recons_losses_fold": val_recons_losses_fold,
            "val_kld_losses_fold": val_kld_losses_fold,
            "metrics_val_results": val_losses_0,
            "val_results": val_vae_losses_1 ,
            "val_recons_results": recons_val_losses_2,
            "val_kld_results": kld_val_losses_3,
            "metrics_test_results": test_losses_0,
            "test_results": test_vae_losses_1,
            "test_recons_results": recons_test_losses_2,
            "test_kld_results": kld_test_losses_3,
        })

    return models, fold_results








