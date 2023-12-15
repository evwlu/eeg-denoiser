# EEG Data Denoising Using Deep Learning

## Introduction

This repository contains the implementation of deep learning models, specifically autoencoders and Conditional Variational Autoencoders (CVAE), for denoising Electroencephalogram (EEG) data. These models aim to improve the quality of EEG data by effectively reducing noise, thereby enhancing the accuracy of brain disorder diagnoses and studies.

## Repository Structure

The repository is structured as follows:

- `__pycache__/`: Compiled Python files.
- `models/`: Contains the implementation of the deep learning models used for denoising.
  - `autoencoder.py`: The autoencoder model.
  - `baseline.py`: Baseline model for comparison.
  - `cvae.py`: Conditional Variational Autoencoder model.
  - `vae.py`: Variational Autoencoder model.
- `outputs/`: Contains the notebooks detailing the training process of the deep learning models used for denoising.
   - `autoencoder.ipynb`: Jupyter notebook detailing the training process of the autoencoder model.
   - `vae.ipynb`: Jupyter notebook detailing the training process of the VAE model.
   - `cvae.ipynb`: Jupyter notebook detailing the training process of the CVAE model.
- `visualizations/`: Contains the scripts and images related to model visuals.
   - `visualization.py`: Utility script for visualizing model results and data.
   - `autoencoder_diagram.png`, `cvae_diagram.png`, `vae_diagram.png`: Visual representations of the model architectures.
- `baseline.ipynb`: Notebook for training and evaluating the baseline model.
- `preprocess.py`: Script for preprocessing EEG data.

- `.gitignore`: Specifies files to be ignored in Git version control.
- `README.md`: Provides an overview and instructions for this repository.

## Getting Started

### Prerequisites

Ensure Python 3.6+ is installed on your system. Dependencies are managed on a per-notebook basis.

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-repo/eeg-denoiser.git
   ```
2. Navigate to the cloned directory:
   ```
   cd eeg-denoiser
   ```

### Data Preparation

Use `preprocess.py` to prepare your EEG datasets. The script will format the data into suitable frames for training.

## Usage

### Training Models

Open and run the Jupyter notebooks (`autoencoder.ipynb`, `baseline.ipynb`) to see the data from the respective models.

### Evaluating Models

The evaluation of each model is included in their respective notebooks. Run these notebooks to assess model performance.

## Visualization

Utilize `visualization.py` to visualize the results of the models, including performance metrics and model architecture diagrams.

## Credits

This project was created by Evan Lu (elu14), Siming Feng (sfeng22), and Ze Hua Chen (zchen186)
