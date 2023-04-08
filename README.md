# Non-blind deconvolution of satellite images using preprocessing, convolutional neural network and custom loss function

### Abstract

The deconvolution of blurred satellite images is a recurring issue whose convolutional neural networks (CNN) have proven effective. In this repository, we combine an initial deterministic deconvolution method (Richardson-Lucy) and a neural network trained with a custom loss function. The aim is deconvolving images taken by the satellite PLEIADES NEO. Results show that our network partly recovers the lost information but give room for improvements.

### Organization of the repository

Two files contain all the code : *process* and *game*. In the first one, you will find functions that import images, create a dataset, define new networks and train them. If you have a very powerful computer and time to spend, you can create a new dataset and retrain the networks. If you only want to visualize our results, you do not need that file. In the *game* file is coded the *pygame* interface. To see it, just execute *game*.

In the data folder, you will find the PSF function of the PLEIADES NEO satellite, three of the fourty $4000\times4000$ satellite images we used and the corresponding $300\times300$ test patches. The storage of a free Github repository is limited to 1Gb, so we could not push the fourty images.

### Visualizing results

1. Clone the repository by typing in your terminal : *git clone https://github.com/XavierMinesParis/deconvolution.git*
2. Be sure that all classical necessary modules are installed : numpy, pygame, torch, cv2, time, etc. If not, an error will occur and tell you which module needs to be installed.
3. Execute the *game.py* file in your favorite Python editor or directly in your terminal : *python game.py*
