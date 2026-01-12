# Sampling with MCGDiff

This repository presents the work we produced at our 3rd year of ENSAE for the sotchastic monte carlo class. It is a reimplementation of the original paper : "Monte Carlo guided Diffusion for Bayesian linear inverse problems",  published by: Gabriel Cardoso, Yazid Janati El Idrissi, Sylvain Le Corff, Eric Moulines. Check here for the original paper : https://arxiv.org/abs/2308.07983


# Organization of the repo

The main part of the code is the MCGDiff.py file that contains the main algorithm of the paper manualy implemented. The notebook presents experience we did to test this code : 

- main_MNIST.ipynb : Contains results of MCGDifff run on small image, using a pre trained DDPM
- GaussianDDPM.ipynb:  contains a simple exmaple of DDPM coded with the posterior gaussian closed form
Other outdated notebooks can be found in the archive folder. 
