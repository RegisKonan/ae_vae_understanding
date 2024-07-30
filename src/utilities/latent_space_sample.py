import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms.transforms import ToPILImage
import sys
sys.path.append('./src')

###### AE Latent space ##########

def latent_space_AE_model(dataset_name,model, test_loader, device, supervised,ax=None):
    model.eval()
    z_latent = []
    labels = []

    for data in test_loader:
        if supervised:  # If test_loader yields tuples containing both images and labels
            img, lbl = data
        else:  # If test_loader only yields images
            img = data
            lbl = None
        
        x_in = img.view(img.size(0), -1).to(device)
        with torch.no_grad():
            z, x_out = model(x_in)

        z_latent.append(z.cpu().numpy())
        if lbl is not None:
            labels.append(lbl.numpy())

    z_latent = np.concatenate(z_latent)
    if len(labels) > 0:
        labels = np.concatenate(labels)

    # Create a colormap for different colors
    if len(np.unique(labels)) > 1:
        colormap = plt.cm.get_cmap('tab10', len(np.unique(labels)))
    else:
        colormap = 'b'

    # Figure with one subplot or use the provided axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # Scatter plot for z_mean
    if supervised:
        scatter_mean = ax.scatter(z_latent[:, 0], z_latent[:, 1],c=labels, cmap=colormap)
    else:
        scatter_mean = ax.scatter(z_latent[:, 0], z_latent[:, 1], c=colormap)
    ax.set_xlabel("Latent Space 1")
    ax.set_ylabel("Latent Space 2")
    ax.set_title(f'Latent Space (z_size = {model.D_z})')
    if len(np.unique(labels)) > 1:
        cbar_mean = plt.colorbar(scatter_mean)
    if dataset_name == "FashionMNIST":
    # Define class labels corresponding to numbers for FashionMNIST
        class_labels = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        # Define class labels on the colorbar
        cbar_mean.set_ticks(list(class_labels.keys()))
        cbar_mean.set_ticklabels(list(class_labels.values()))
    
    plt.tight_layout()

    # If no axis, show the plot
    if ax is None:
        plt.show()


##### VAE Latent space ########
def latent_space_VAE_model(dataset_name,model, test_loader, device, number_fold, kl_div, gaussian_distribution,supervised):

    model.eval()
    latent_variables = []
    labels = []

    for data in test_loader:
        if supervised:  # If test_loader yields tuples containing both images and labels
            img, lbl = data
        else:  # If test_loader only yields images
            img = data
            lbl = None
        x_in = img.view(img.size(0), -1).to(device)
        with torch.no_grad():
            z_mean, z_logvar = model.encode(x_in)
            z = model.reparameterization(z_mean, z_logvar)
        latent_variables.append(z.cpu().numpy())
        if lbl is not None:
            labels.append(lbl.numpy())
    
    latent_variables = np.concatenate(latent_variables)
    if len(labels) > 0:
        labels = np.concatenate(labels)
    
    if len(np.unique(labels)) > 1:
        colormap = plt.cm.get_cmap('tab10', len(np.unique(labels)))

    else:
        colors= 'b'        

    # Create a figure with three subplots arranged horizontally
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    

    # Scatter plot for latent space on the left
    # scatter_latent_variables = ax1.scatter(latent_variables[:, 0], latent_variables[:, 1], c=labels, cmap=colormap)
    if supervised:
        scatter_latent_variables = ax1.scatter(latent_variables[:, 0], latent_variables[:, 1],c=labels, cmap=colormap)
    else:
        scatter_latent_variables= ax1.scatter(latent_variables[:, 0], latent_variables[:, 1], c=colors)
    ax1.set_xlabel("Latent Space 1")
    ax1.set_ylabel("Latent Space 2")
    ax1.set_title(f'Latent Space (z_size = {model.D_z}) for fold {number_fold}')
    if len(np.unique(labels)) > 1:
        cbar_mean = plt.colorbar(scatter_latent_variables)
    
    if kl_div == "standard_gaussian":  
        # Plot the distribution of the first latent variable (z[0]) and the second latent variable (z[1]) 
        ax2.hist(latent_variables[:, 0], bins=50, density=True, alpha=0.5, color='blue', label='Latent Space 1')
        ax3.hist(latent_variables[:, 1], bins=50, density=True, alpha=0.5, color='blue', label='Latent Space 2')

        # Overlay the standard normal distribution (mean=0, std=1) using an orange line
        x = np.linspace(-5, 5, 1000)
        y = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)
        ax2.plot(x, y, color='orange', label='Standard Normal Distribution')
        ax3.plot(x, y, color='orange', label='Standard Normal Distribution')

        ax2.set_xlabel("Latent Space")
        ax2.set_ylabel("Density")
        ax2.set_title(f'Distribution of Latent Space 1 (z_size = {model.D_z}) for fold {number_fold}')
        ax2.legend()

        ax3.set_xlabel("Latent Space")
        ax3.set_ylabel("Density")
        ax3.set_title(f'Distribution of Latent Space 2 (z_size = {model.D_z}) for fold {number_fold}')
        ax3.legend()

    elif kl_div == "mean_conditional_gaussian":
        
        # Plot the distribution of the first latent variable (z[0]) and the second latent variable (z[1]) 
        ax2.hist(latent_variables[:, 0], bins=50, density=True, alpha=0.5, color='blue', label='Latent Space 1')
        ax3.hist(latent_variables[:, 1], bins=50, density=True, alpha=0.5, color='blue', label='Latent Space 2')

        # Overlay the normal distribution with mean=mu, std=sigma using an orange line
        mu1, mu2 = gaussian_distribution['mean'].numpy()[0]
        sigma1, sigma2 = gaussian_distribution['std'].numpy()[0]
        
         # Accessing x1 and x2
        x1_values = gaussian_distribution['x1']
        x2_values = gaussian_distribution['x2']
        y1 = (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1_values  - mu1) / sigma1) ** 2)
        y2 = (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1_values  - mu2) / sigma2) ** 2)

        ax2.plot(x1_values , y1, color='orange', label=f'Normal Distribution (mu={mu1}, sigma={sigma1**2})')
        ax3.plot(x2_values , y2, color='orange', label=f'Normal Distribution (mu={mu2}, sigma={sigma2**2})')

        ax2.set_xlabel("Latent Space")
        ax2.set_ylabel("Density")
        ax2.set_title(f'Distribution of Latent Space 1 (z_size = {model.D_z}) for fold {number_fold}')
        ax2.legend()

        ax3.set_xlabel("Latent Space")
        ax3.set_ylabel("Density")
        ax3.set_title(f'Distribution of Latent Space 2 (z_size = {model.D_z}) for fold {number_fold}')
        ax3.legend()

    else:
        raise ValueError("Invalid KLD loss type. Supported types are 'variational_inference' and 'uncorrelated_Gaussians'.")
        
    plt.tight_layout()
    
    if dataset_name == "FashionMNIST":
        # Define class labels corresponding to numbers for FashionMNIST
        class_labels = {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot"
        }

        # Define class labels on the colorbar
        cbar_mean.set_ticks(list(class_labels.keys()))
        cbar_mean.set_ticklabels(list(class_labels.values()))

        
    # Return the generated figure instead of showing it
    return fig



################ generate data from AE latent space ##################

def generate_data_from_AE_latent_space(model, D_z, img1,img2,image_to_sample=1, nrow=10):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = torch.randn((image_to_sample, D_z)).to(device)
        samples = model.decoder(z).cpu().view(image_to_sample, 1,img1, img2)
        grid = make_grid(samples, nrow=nrow)

        # Convert image to RGBA format and set background to white
        image = ToPILImage()(grid)
        image = image.convert("RGBA")
        datas = image.getdata()
        new_data = []
        for item in datas:
            # Set non-white pixels to white
            if item[0] < 100 and item[1] < 100 and item[2] < 100:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        image.putdata(new_data)
        image = image.convert("RGB")

        return image

################ generate data from VAE latent space ##################

def generate_data_from_VAE_latent_space(model, gaussian_distribution, D_z, image_to_sample, nrow,img1,img2,kl_div):
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("KL Div Loss Type:", kl_div) 
        if kl_div == "standard_gaussian":
            z = torch.randn((image_to_sample, D_z)).to(device) 
        elif kl_div == "mean_conditional_gaussian":
            z = torch.randn((image_to_sample, D_z)).to(device) * gaussian_distribution['std'] + gaussian_distribution['mean']
        else :
             raise ValueError("Invalid KLD loss type. Supported types are 'standard_gaussian' and 'mean_conditional_gaussian'.")

        samples = model.decoder(z).cpu().view(image_to_sample, 1, img1, img2)  
        grid = make_grid(samples, nrow=nrow)

        # Convert image to RGBA format and set background to white
        image = ToPILImage()(grid)
        image = image.convert("RGBA")
        datas = image.getdata()
        new_data = []
        for item in datas:
            # Set non-white pixels to white
            if item[0] < 100 and item[1] < 100 and item[2] < 100:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        image.putdata(new_data)
        image = image.convert("RGB")

        return image
    
    ######################### Exploration of the Latent Space #####################################
    

def visualize_latent_space(model, device,img1,img2):
    z1_range = np.linspace(-3, 3, 21)
    z2_range = np.linspace(-3, 3, 21)
    results = []

    with torch.no_grad():
        for z1 in z1_range:
            for z2 in z2_range:
                results.append(model.decoder(torch.tensor([[z1, z2]], device=device, dtype=torch.float32)).view(-1, 1, img1, img2))

    image = ToPILImage()(make_grid(torch.cat(results, dim=0), 21))

    # Convert image to RGBA format and set background to white
    image = image.convert("RGBA")
    datas = image.getdata()
    new_data = []
    for item in datas:
        # Set non-white pixels to white
        if item[0] < 100 and item[1] < 100 and item[2] < 100:
            new_data.append((255, 255, 255, 50))
        else:
            new_data.append(item)
    image.putdata(new_data)
    image = image.convert("RGB")

    plt.imshow(image)
    plt.xticks([2 + (img1/2) + (img1+2) * ii for ii in range(len(z1_range))], [f'{x:.1f}' for x in z1_range])
    plt.yticks([2 + (img2/2) + (img2+2) * ii for ii in range(len(z2_range))], [f'{x:.1f}' for x in z2_range])
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.show()
    
    return image

