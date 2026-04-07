# XAI_FinalAssignment
Final XAI assignment for Bart van Wees (s1146935) implementing GradCAM and Smoothgrad.

# Explainable AI – Robustness of Grad-CAM and SmoothGrad on CIFAR-10

This project investigates the robustness of the explanation methods Grad-CAM and 
SmoothGrad, applied to a CNN image classifier trained on CIFAR-10. Explanations are 
compared before and after small input perturbations (Gaussian noise and translation) 
using Pearson correlation as a stability metric.

# Reproducing the experiment
## Requirements

- Python 3.11
- torch 2.10.0
- torchvision 0.25.0
- numpy 1.26.4
- matplotlib 3.10.8
- scipy 1.17.1

## Setting up the environment and Running the Code
Run the following code in the Anaconda Prompt Terminal:

1. Create the environment: `conda create -n xai python=3.11`
2. Activate the environment:`conda activate xai`
3. Install required dependencies: `pip install torch==2.10.0 torchvision==0.25.0 numpy==1.26.4 matplotlib==3.10.8 scipy==1.17.1`
4. Install Jupyter Lab: `pip install jupyterlab`
5. Open A jupyter notebook: `jupyter lab`

Inside the notebook, do the following:
1. Navigate and open: `XAI_FinalAssignment_BartvanWees.ipynb`
2. Run all cells in order from top to bottom

The notebook will automatically download the CIFAR-10 dataset on first run. The model will train for 25 epochs, which might take a short while, depending on your system. To skip training, the model weights are saved to `cifar_cnn.pth` after the first run and loaded automatically. This file is also present in the repository, allowing the training step to be skipped.
