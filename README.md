# ae_vae_understanding

Work by Regis Djaha and Iñigo Urtega on understanding AEs and VAEs.

[Regis Djaha ][Regis], [Iñigo Urteaga][Iñigo ]

[Regis]: https://github.com/RegisKonan
[Iñigo]: https://iurteaga.github.io/

This work lead to this Master Thesis [project](https://github.com/RegisKonan/ae_vae_understanding/blob/a00e1b481b5d796bc9fea429763acd4923cc4698/reports/Regis_Djaha_AMMI_project_2023.pdf).

[paper]: ...

## Repository structure

### doc

Directory for documentation (reports, papers of interest, etc).

### src

Directory where our code will be place in.
## Installation

1. Clone this repository.
   ```sh
    git clone https://github.com/RegisKonan/ae_vae_understanding.git
    cd ae_vae_understanding
   ```

2. Install package virtualenv
   ```sh
   pip install virtualenv
   ```
3. Create a environment python
   ```sh
   virtualenv [name], example :  virtualenv env
   ```
4. Activate virtualenv
   
```sh 
macOS version :   source [name]/bin/activate, example: source env/bin/activate
```

```sh 
windows version :  \[name]\Scripts\activate, example: \env\Scripts\activate
```
5. Install `requirements.txt`.

 ```sh
 pip install -r requirements.txt
 ```
## By using conda
1.  
   ```sh
   conda create -n [name], example: conda create -n env or conda create -n env python=3.12.2
   ```
2. Activate 
```sh
conda activate [name], example: conda activate env
```
3. Install `requirements.txt`.
```sh
pip install -r requirements.txt
```
   
## Running our experiments

1. Make sure you are in the main `ae_vae_understanding` directory

```
cd /path-to-ae_vae_understanding
```
   
### AEs Experiments and VAEs Experiments

You can choose the type of dataset and model you want to use, as well as the number $n_L$ of layers and dimensions $D_z$.

```sh

dataset_name= [MNIST, FashionMNIST, FreyFace]

model_name = [AE, VAE]

```

To run the experiment scripts, call them from your `vae_understanding` directory:

1. Test1: [script/AE_VAE_experiment_nL.py](./script/AE_VAE_experiment_nL.py).
   
```sh
python script/AE_VAE_experiment_nL.py -dataset_name FreyFace -model_name AE   -n_L 3 7  -D_z 5
```

In this script, we have fixed the dimension $D_z$ and varied the number of layers.


2. Test2: [script/AE_VAE_experiment_Dz.py](./script/AE_VAE_experiment_Dz.py).
   
```sh
python script/AE_VAE_experiment_Dz.py -dataset_name FashionMNIST -model_name VAE   -n_L 3   -D_z  2 5
```

In this script, we have fixed the number of layers and varied the dimensions.

### VAE Prior Experiments

We used the same code as in [script/AE_VAE_experiment_nL.py](./script/AE_VAE_experiment_nL.py) and [script/AE_VAE_experiment_Dz.py](./script/AE_VAE_experiment_Dz.py), replacing standard_gaussian with mean_conditional_gaussian and defining gaussian_distribution as gaussian_distribution.

In this experiment, we defined means and standard deviations.

### $\beta$-VAE Experiments

1. Test: [script/beta_VAE_experiment.py](./script/beta_VAE_experiment.py) 
   
```sh
python script/AE_VAE_experiment.py -dataset_name MNIST -model_name VAE  -beta 0.02 1 1.5 10 -n_L 1 -D_z 5
```

In this script, we have fixed the value of beta and varied the number of layers and the dimension $D_z$
​
 . You don't need to specify a model, as it is already defined in the code.

### Run plot metrics, latent space, original and reconstructions images and number of parameter.

1. Test1: [script/Run_code_AE_VAE.py](./script/Run_code_AE_VAE.py) 
   
```sh
python script/Run_code_AE_VAE.py -dataset_name FreyFace -model_name AE  -n_L 1 -D_z 2
```


2. Test2: [script/Run_code_beta_VAE.py](./script/Run_code_beta_VAE.py) 
   
```sh
python script/Run_code_beta_VAE.py -dataset_name FreyFace -model_name VAE  -beta 0.02 1 1.5 10 -n_L 1 -D_z 5
```

## Notebooks

Notebooks are included to show the followed steps.

1. Start [jupyter](https://jupyter.org).
```sh
   jupyter notebook
```

1. Navigate to the [notebooks](https://github.com/iurteaga/vae_understanding/tree/0d6386509fea6a3bf58298143d00c1475ddfb2df/notebooks) folder.

1. First run all cells.

 
Questions and Issues
--------------------

If you find any bugs or have any questions about this code, please contact Regis [rdjaha@bcam.org](rdjaha@bcam.org). 

Acknowledgments
--------------------

This project funded by La Caixa Junior Leader.
