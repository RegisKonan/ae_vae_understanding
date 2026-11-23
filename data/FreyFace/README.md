# FreyFace dataset

The **Frey Face** dataset contains about **2,000 grayscale images of Brendan Frey's face**, taken from sequential frames of a short video. Each image is **20 × 28 pixels**. It is a classic benchmark for generative models and was used in the original VAE paper (Kingma & Welling, 2014).

## File

| File | Description |
|---|---|
| `frey_rawface.mat` | Raw images, in MATLAB format |

**Source:** Sam Roweis' data page, <https://cs.nyu.edu/~roweis/data/frey_rawface.mat>

## Usage

The file is loaded by `FreyFaceLoader` in [`src/main/datasets.py`](../../src/main/datasets.py). Keep it in `data/FreyFace/` and select it with:

```bash
python script/AE_VAE_experiment_nL.py -dataset_name FreyFace -model_name VAE -n_L 2 -D_z 2
```

## Other datasets

**MNIST** (handwritten digits) and **Fashion-MNIST** (clothing items) are downloaded automatically by `torchvision` the first time they are used, so they are not stored in this repository.
