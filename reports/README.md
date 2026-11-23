# Reports

| File | Description |
|---|---|
| [`Regis_Djaha_AMMI_project_2023.pdf`](Regis_Djaha_AMMI_project_2023.pdf) | Master's project report (AMMI, 2023), *Understanding Autoencoders and Variational Autoencoders* |
| [`ML_Palencia_Presentation_new.key`](ML_Palencia_Presentation_new.key) | Presentation slides (Apple Keynote) |

## Understanding Autoencoders and Variational Autoencoders

**Regis Konan Marcel Djaha**, with Iñigo Urteaga, Basque Center for Applied Mathematics (BCAM)

This report explores **autoencoders (AEs)**, **variational autoencoders (VAEs)** and **β-VAEs**, three fundamental unsupervised learning methods. It covers:

- **Theory:** how AEs learn deterministic latent representations; how VAEs learn a probabilistic latent space by maximising the ELBO; and how β-VAEs reweight the KL term.
- **Implementation:** encoder–decoder architectures with a variable number of layers and latent dimensions, in PyTorch.
- **Experiments on MNIST, Fashion-MNIST and FreyFace:** the impact of network depth, latent dimension, prior distribution and β on training losses, reconstructions, latent-space structure and data generation.

These results clarify what AEs and VAEs are good at (data representation, dimensionality reduction, image generation and latent-space exploration) and how to choose their hyperparameters in practice.

*This work was supported by the "la Caixa" Junior Leader programme (award LCF/BQ/PI22/11910028).*
