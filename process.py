# -*- coding: utf-8 -*-
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import math as m
from numpy.fft import fftshift, ifftshift, fft2, ifft2, fft, ifft
from skimage import color, data, restoration
from scipy.signal import wiener, convolve2d, convolve
from scipy import ndimage
import cv2
from glob import glob
import shutil
import random
from torch.utils.data import Dataset
from time import sleep
import requests
plt.rcParams["figure.figsize"] = (6, 6)
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Conv2d, ReLU, Module, MaxPool2d, Upsample, BatchNorm2d
import cProfile, pstats
import yaml
import copy
import pygame as pg
import random as rd
from skimage import io
import h5py
Gris = (150, 150, 150)
Gris_fonce = (80, 80, 80)
Gris_clair = (200, 200, 200)
Blanc = (255, 255, 255)
Noir = (0, 0, 0)
Or = (205, 155, 0)
Eau = (100, 120, 240)
Orange_clair = (230, 170, 30)
Orange = (200, 110, 15)
Orange_fonce = (160, 80, 0)
Corail = (240, 100, 0)
Cafe = (100, 30, 15)
Violet = (120, 0, 150)
Bleu_pale = (60, 200, 200)
Bleu_clair = (160, 160, 240)
Bleu_fonce = (25, 50, 240)


class Generation():
    """
    The Generation class can build the whole dataset :
    - importing the global images from the geo2france server
    - extracting and saving the local patches
    """
    
    def import_images(n_images:int, start_ind:int = 0) -> None:
        """
        Downloads n_images random 4000x4000 patches from geo2france geoserver on the drive

        Attributes :
            n_images (int) : number of random images to download
            start_ind (int) : start index for the name of the files (useful to download images in several steps)
        Returns : None
        """
        Xmin, Ymin, Xmax, Ymax = 1716008.81, 8289759.24, 1724688.82, 8298456.62
        errors = 0
        for i in range(n_images):
            x_min = np.random.randint(int(Xmin), int(Xmax)-400) + 1e-2 * np.random.randint(0, 100)
            y_min = np.random.randint(int(Ymin), int(Ymax)-400) + 1e-2 * np.random.randint(0, 100)
            url = f"https://www.geo2france.fr/geoserver/agglo_st_quentin/"
            url += "ows?SERVICE=WMS&REQUEST=GetMap&CRS=EPSG:3949&LAYERS=agglo_st_quentin_ortho_2012_"
            url += "vis&BBOX={x_min},{y_min},{x_min + 400},{y_min + 400}&WIDTH=4000&HEIGHT=4000&FORMAT=image/geotiff"
            try:
                img_data = requests.get(url).content
            except:
                errors +=1
            with open(f"data/satellite_images/{i + start_ind}.tif", "wb") as handler:
                handler.write(img_data) 
            sleep(10)
        print(f"Done with {errors} errors")
        
    def generate_dataset(sample_size = 300, n_patches = 10, train = 0.7, val = 0.1, test = 0.2):
        """
        Generates the main dataset of the project. A number of sample_size patches are taken from the satellite images
        and are saved into three folders : train, validation and test. It is adviced (but not necessary) to choose
        n_patches grader than 100vand train, val and test ratios with a sum of 1.
        
        Attributes:
        sample_size (int) : size of the patches. 300 is adviced because it is larger enough to see structures on the patch
        and remains computationally sober.
        n_patches (int) : number of patches
        train (float) : proportion of dataset used for training the model
        val (float) : proportion of dataset used for the validation steps
        test (float) : proportion of dataset used for testing the model 
        """
        if os.path.exists("data/train/"): #building the global structure of the folders
            shutil.rmtree("data/train/")
        if os.path.exists("data/val/"):   #data --- train --- labels --- 0.0.0.npy
            shutil.rmtree("data/val/")    #      |         |          |- 0.0.1.npy
        if os.path.exists("data/test/"):  #      |         |          |- etc.
            shutil.rmtree("data/test/")   #      |         |
        os.mkdir("data/train/")           #      |         |- images --- etc.
        os.mkdir("data/val/")             #      |         |
        os.mkdir("data/test/")            #      |         |- RLimages --- etc.
        os.mkdir("data/train/labels/")    #      |
        os.mkdir("data/train/images/")    #      |- val --- etc.
        os.mkdir("data/train/RLimages/")  #      |
        os.mkdir("data/train/mix_images/")#      |
        os.mkdir("data/val/labels/")      #      |- test --- etc.
        os.mkdir("data/val/images/")      #      |    
        os.mkdir("data/val/RLimages/")    #      |- satellite_images --- 1.tif
        os.mkdir("data/val/mix_images/")  #                           |
        os.mkdir("data/test/labels/")     #                           |- 2.tif
        os.mkdir("data/test/images/")     #                           |- etc.
        os.mkdir("data/test/RLimages/")
        os.mkdir("data/test/mix_images/")
        
        psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt") # loading the PLEIADES NEO PSF
        psf = psf / psf.sum() # normalizing the psf
        
        images = glob("data/satellite_images/*.tif") # loading the paths to the 4000x4000 satellite images
        if len(images) < 10:
            raise Exception("An amount of at least 10 satellite images is necessary.\nPlease use the import_images function to download them.")
        random.shuffle(images) # shuffling the images
        n_images = len(images)
        train_images = images[: int(train*n_images)] # the images whose the patches will be extracted
        val_images = images[int(train*n_images): int(train*n_images) + int(val*n_images)]
        test_images = images[int(train*n_images) + int(val*n_images): ]
        n_patches_per_image = int(n_patches/n_images) + 1 # number of patches per image
        
        folders, packs = ["train", "val", "test"], [train_images, val_images, test_images]
        # the train, validation and test sets will be built on distinct images (for independency purpose)
        for folder, pack in zip(folders, packs):
            for image in pack:
                num_image = image[len(f"data/satellite_images/"): -4]
                print("Extracting patches from", image)
                im = cv2.imread(image).mean(axis=2) # loading the 4000x4000 satellite image
                im = im / im.max() # normalization
                h, w = im.shape
                n, m = (h-30)//sample_size, (w-30)//sample_size # number of patches for each side of the image
                possibilities = [(i, j) for i in range(n) for j in range(m)]
                choices = random.sample(possibilities, n_patches_per_image)
                for couple in choices:
                    i, j = couple # (num_image, i, j) is the index of a patch
                    x, y = i*sample_size + 15, j*sample_size + 15
                    patch = im[x: x + sample_size, y: y + sample_size] # patching the image
                    large_patch = im[x - 15: x + sample_size + 15, y - 15: y + sample_size + 15]
                    np.save(f"data/{folder}/labels/{num_image}.{i}.{j}.npy", np.expand_dims(patch, axis=0))
                    
                    new_psf = np.pad(psf, (large_patch.shape[0] - psf.shape[0]) // 2) # padding
                    O, H = fft2(large_patch), fft2(new_psf) # getting the fourier transforms once the size are the same
                    new_patch = np.real(ifftshift(ifft2(O * H))) # convolution with psf
                    # down resolution and adding gaussian noise
                    new_patch = cv2.resize(new_patch[1: : 3, 1: : 3], large_patch.shape, interpolation = cv2.INTER_CUBIC)
                    # adding noise, its standard deviation is equal to the label's mean divided by 90 (usual ratio)
                    new_patch = new_patch + np.random.normal(size = new_patch.shape, scale = patch.mean()/90)
                    np.save(f"data/{folder}/images/{num_image}.{i}.{j}.npy",
                            np.expand_dims(new_patch[15: -15, 15: -15], axis=0))
                    
                    #Richardson-Lucy preprocessing
                    RL_patch = restoration.richardson_lucy(new_patch, psf, num_iter=30)[15: -15, 15: -15]
                    patch = new_patch[15: -15, 15: -15]
                    np.save(f"data/{folder}/RLimages/{num_image}.{i}.{j}.npy",
                            np.expand_dims(RL_patch, axis=0))
                    
                    #mix input preprocessing
                    FFT_input = fftshift(fft2(patch))
                    FFT_input2 = fftshift(fft2(RL_patch))
                    new_FFT_input = np.zeros((sample_size, sample_size), dtype='complex')
                    new_FFT_input[sample_size//3: 2*sample_size//3,
                                  sample_size//3: 2*sample_size//3] = FFT_input[sample_size//3: 2*sample_size//3,
                                                                                sample_size//3: 2*sample_size//3]
                    FFT_input2[sample_size//3: 2*sample_size//3, sample_size//3: 2*sample_size//3] = 0
                    mix_FFT = fftshift(new_FFT_input + FFT_input2)
                    mix_patch = np.real(ifft2(mix_FFT))
                    np.save(f"data/{folder}/mix_images/{num_image}.{i}.{j}.npy", np.expand_dims(mix_patch, axis=0))



class Processing():
    """
    The Processing class aims at providing useful analysis tools like metrics calculation or PSD computation.
    """
    
    def PSNR(output:torch.tensor, target:torch.tensor) -> torch.tensor:
        """
        Computes the Peak Signal to Noise Ratio (PSNR) between output image from CNN and clean image
        (source : https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio)

        Parameters :
            output (torch.tensor) : Output image of the CNN
            target (torch.tensor) : Clean image
        Returns :
            torch.tensor : 0D tensor containing the value of the PSNR
        """
        
        mse =  nn.MSELoss(reduction='mean')(output, target)
        return 10*torch.log(((target.max())**2)/mse)
    
    def custom_loss(norm_coeff:float = 0, angle_coeff:float = 0, energy_coeff:float = 1,
                    var_coeff:float = 0, abs_coeff:float = 0, cross_coeff:float = 0):
        """
        Returns the custom loss function with given importances

        Parameters :
            norm_coeff (float) : Coefficient of the L1 distance in norm in Fourier transform
            angle_coeff (float) : Coefficient of the L1 distance in phase in Fourier transform
            energy_coeff (float) : Coefficient of the L2 distance between the two images (MSE loss)
            var_coeff (float) : Coefficient of the L2 distance between the gradients of the two images
            abs_coeff (float) : Coefficient of the L1 distance
            cross_coeff (float) : Coefficient of the cross-correlation between the two images
        Returns :
            function : the custom loss function
        """
        
        def res(output, target):
            norm_distance, energy_distance, angle_distance, tot_var_distance, abs_distance = 0, 0, 0, 0, 0
            cross_distance = 0
            xtfd = torch.fft.fft2(output)  # fourier transform
            ytfd = torch.fft.fft2(target)
            
            if norm_coeff != 0:
                xnorm, ynorm = torch.abs(xtfd), torch.abs(ytfd)
                norm_distance = norm_coeff*nn.L1Loss()(xnorm, ynorm)
                
            if angle_coeff != 0:
                xangle, yangle = torch.angle(xtfd), torch.angle(ytfd)
                angle_distance = angle_coeff*nn.L1Loss()(xangle, yangle)
                
            if energy_coeff != 0:
                energy_distance = 1000*energy_coeff*nn.MSELoss()(output, target)
                
            if var_coeff != 0:
                grad = torch.gradient(output[: , 0] - target[: , 0])
                gradx, grady = grad[1], grad[2]
                norm_grad = torch.sqrt(gradx**2 + grady**2)
                tot_var_distance = 100*var_coeff*torch.mean(norm_grad)
                
            if abs_coeff != 0:
                abs_distance = 100*abs_coeff*nn.L1Loss()(output, target)
                
            if cross_coeff != 0:
                cross_distance = 10**(-2)*cross_coeff*nn.CrossEntropyLoss()(np.squeeze(output.cpu()), np.squeeze(target.cpu()))
            
            denominator = norm_coeff + angle_coeff + energy_coeff + var_coeff + abs_coeff + cross_coeff
            return (norm_distance + energy_distance + angle_distance + tot_var_distance + abs_distance + cross_distance)/denominator
        return res
    
    def wiener_filter(label, noverlap = 20, std = None):
        """
        Returns an image restored the Wiener filter

        Parameters :
            label (numpy.ndarray) : Target image we want the filter to be based on
            noverlap (int) : size of the window for Welch's method
            std (float) : Noise ratio 
        Returns :
            numpy.ndarray : The Wiener filter kernel
        """
        if std == None: # default value for noise ratio is 90
            std = label.mean()/90
            
        psf = np.loadtxt("data/psf/psf_pan_2d_30cm_from10cm.txt")
        psf = psf / psf.sum() # normalizing the PSF
        return restoration.wiener(label, psf, 1)

    def PSD(image, nperseg = 100, noverlap = 10):
        if noverlap == None:
            noverlap = nperseg//2
        n = len(image)
        N = (n - nperseg)//noverlap + 1
        h = np.blackman(nperseg)
        h = np.atleast_2d(h).T @ np.atleast_2d(h)
        new_image = image - np.mean(image)
        PSD_matrix = np.zeros((nperseg, nperseg))
        for i in range(N):
            for j in range(N):
                patch = new_image[i*noverlap: i*noverlap + nperseg, j*noverlap: j*noverlap + nperseg]
                PSD_matrix += fftshift(np.abs(fft2(h*patch))**2)
        PSD_matrix = PSD_matrix / N**2
        return PSD_matrix

    def cross_PSD(image1, image2, nperseg = 100, noverlap = 10):
        if noverlap == None:
            noverlap = nperseg//2
        n = len(image1)
        N = (n - nperseg)//noverlap + 1
        h = np.blackman(nperseg)
        h = np.atleast_2d(h).T @ np.atleast_2d(h)
        new_image1 = image1 - np.mean(image1)
        new_image2 = image2 - np.mean(image2)
        PSD_matrix = np.zeros((nperseg, nperseg))
        for i in range(N):
            for j in range(N):
                patch1 = new_image1[i*noverlap: i*noverlap + nperseg, j*noverlap: j*noverlap + nperseg]
                patch2 = new_image2[i*noverlap: i*noverlap + nperseg, j*noverlap: j*noverlap + nperseg]
                PSD_matrix += np.real(fftshift(fft2(h*patch1) * np.conjugate(fft2(h*patch2))))
        PSD_matrix = PSD_matrix / N**2
        return PSD_matrix
    
    def save_cross_psd(name, image1, image2, nperseg = 100, noverlap = 10, super_reso = False, hyper_reso = False):
        rapport = Processing.cross_PSD(image1, image2)/np.sqrt(Processing.PSD(image1)*Processing.PSD(image2)+0.01)
        plt.figure()
        plt.imshow(rapport, vmin=0, vmax=1)
        plt.colorbar()
        centre = nperseg//2
        plt.plot([nperseg//3, 2*nperseg//3, 2*nperseg//3, nperseg//3, nperseg//3],
                 [2*nperseg//3, 2*nperseg//3, nperseg//3, nperseg//3, 2*nperseg//3],
                 color='b', linewidth=3, label = "Nyquist Frequency 30 cm (0.017 cm^-1)")
        theta = np.linspace(0, 2*np.pi, 100)
        x, y = 0.41*nperseg*np.cos(theta)/2 + nperseg/2, 0.41*nperseg*np.sin(theta)/2 + nperseg/2
        if hyper_reso:
            plt.plot(x, y, color='r', linewidth=3, label = "Cut-off Frequency 24 cm (0.020 cm^-1)")
        plt.legend()
        plt.savefig("networks/" + name + "/psd.jpg")
        plt.close()
        if super_reso and not hyper_reso:
            rapport[nperseg//3: 2*nperseg//3, nperseg//3: 2*nperseg//3] = 0
            return 9*np.mean(rapport)/8
        if hyper_reso:
            rapport[nperseg//3: 2*nperseg//3, nperseg//3: 2*nperseg//3] = 0
            super_ = 9*np.mean(rapport)/8
            #the hyper-resolution circle
            mask = np.zeros((nperseg, nperseg))
            c = 0
            for i in range(nperseg):
                for j in range(nperseg):
                    if np.sqrt((i-nperseg/2)**2 + (j-nperseg/2)**2) > 0.41*nperseg/2:
                        mask[i, j] = 1
                        c += 1
            rapport[mask == 0] = 0
            hyper_ = 100**2*np.mean(rapport)/c**2
            return super_, hyper_
        
    def quality_grid(n_val = 10, side = 11):
        names = np.array(glob("data/val/labels/*.npy"))[: n_val]
        network_names = os.listdir("grid")
        n_networks = len(network_names)
        networks = []
        for i in range(side):
            networks.append([])
            for j in range(side):
                with open(f"grid/{i-side//2}_{j-side//2}_cnn" + "/history.yaml", 'r') as fichier:
                    history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
                    network = Deconv.load(f"grid/{i-side//2}_{j-side//2}_cnn", branch = False)
                networks[i].append(network)
        rmses, super_resos = np.zeros((side, side)), np.zeros((side, side))
        for k in range(n_val):
            id_name = names[k][len(f"data/val/labels/"): -4]
            label = np.load("data/val/labels/" + id_name + ".npy")[0]
            denominateur = len(label)*np.mean(label)
            PSD_label = Processing.PSD(label)
            input_ = np.load("data/val/images/" + id_name + ".npy")[0]
            input2 = np.load("data/val/RLimages/" + id_name + ".npy")[0]
            for i in range(side):
                for j in range(side):
                    print(k, i, j, end='\r')
                    network = networks[i][j]
                    if network.history['RL']:
                        output_tensor = network(torch.from_numpy(input2).float()[None, None, ...])
                    elif network.history['mix']:
                        output_tensor = network(torch.from_numpy(input3).float()[None, None, ...])
                    else:
                        output_tensor = network(torch.from_numpy(input_).float()[None, None, ...])
                    output = output_tensor.detach().numpy()[0][0]
                    rmses[i, j] += 100*np.sqrt(np.sum((label - output)**2))/denominateur
                    rapport = 100*Processing.cross_PSD(label, output)/np.sqrt(PSD_label*Processing.PSD(output)+0.01)
                    nperseg = 100
                    rapport[nperseg//3: 2*nperseg//3, nperseg//3: 2*nperseg//3] = 0
                    super_resos[i, j] += 9*np.mean(rapport)/8
        rmses, super_resos = rmses/n_val, super_resos/n_val
        plt.figure()
        plt.imshow(rmses)
        plt.colorbar()
        plt.xlabel("Power of cross-entropy coeff")
        plt.ylabel("Power of Fourier angle coeff")
        plt.title(f"nRMSE of the grid for {n_val} validation images")
        plt.savefig("networks/plots/rmse_grid.jpg")
        plt.close()
        plt.figure()
        plt.imshow(super_resos)
        plt.colorbar()
        plt.xlabel("Power of cross-entropy coeff")
        plt.ylabel("Power of Fourier angle coeff")
        plt.title(f"Super-resolution of the grid for {n_val} validation images")
        plt.savefig("networks/plots/super_reso_grid.jpg")
        plt.close()
        plt.figure()
        scatters = []
        for i in range(side):
            for j in range(side):
                color = (0, i/(side-1), j/(side-1))
                scatters.append(plt.scatter(rmses[i, j], super_resos[i, j], marker='o', color=color))
        plt.xlabel("n-RMSE (%)")
        plt.ylabel("Super-resolution (%)")
        plt.savefig("networks/plots/scatter_grid.jpg")
        plt.close()


class MyDataset(Dataset):

    def __init__(self, train = True, val = False, test = False, RL = False, mix = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # we use the GPU if available
        self.train = train
        self.val = val
        self.test = test
        if val:
            self.train, self.test = False, False
            folder = 'val'
        elif test:
            self.train, self.val = False, False
            folder = 'test'
        else:
            folder = 'train'
        self.folder = folder
        self.images = []
        self.labels = []
        self.ids = []
        if RL:
            images = glob(f"data/" + folder + "/RLimages/*.npy")
        elif mix:
            images = glob(f"data/" + folder + "/mix_images/*.npy")
        else:
            images = glob(f"data/" + folder + "/images/*.npy")
        n_train = len(images)
        for name in images:
            print("Opening", name, end ='\r')
            if RL:
                id_name = name[len(f"data/" + folder + "/RLimages/"): -4]
                self.ids.append(id_name)
            elif mix:
                id_name = name[len(f"data/" + folder + "/mix_images/"): -4]
                self.ids.append(id_name)
            else:
                id_name = name[len(f"data/" + folder + "/images/"): -4]
                self.ids.append(id_name)
            image = np.load(name)
            self.images.append(image)
            label = np.load(f"data/" + folder + "/labels/" + id_name + ".npy")
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx:int) -> tuple:
        x = torch.from_numpy(self.images[idx]).float().to(self.device)
        y = torch.from_numpy(self.labels[idx]).float().to(self.device)
        return x, y


class Network(Module):

    def __init__(self, name = "cnn", norm_coeff = 0, angle_coeff = 0,
                 energy_coeff = 0, var_coeff = 0, abs_coeff = 1, cross_coeff = 0):
        super(Network, self).__init__() # calling the Pytorch's Module's constructor
        self.norm_coeff = norm_coeff
        self.angle_coeff = angle_coeff
        self.energy_coeff = energy_coeff
        self.var_coeff = var_coeff
        self.abs_coeff = abs_coeff
        self.cross_coeff = cross_coeff
        self.is_trained = False # we just created the instance, it can't already be trained
        # adding the additional info to the name
        self.name = name
        self.history = {"norm_coeff": norm_coeff, "angle_coeff": angle_coeff, "energy_coeff": energy_coeff,
                       "var_coeff": var_coeff, "abs_coeff": abs_coeff, "cross_coeff": cross_coeff}
        self.weights = []
        self.loss = 0
        self.PSNR = 0
        self.layers = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # defaulting to GPU if available
        if not os.path.exists("networks/"):
            os.mkdir("networks/")

    def forward(self, x:torch.tensor)->torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def fit(self, batch_size:int = 32, epochs:int = 10,
            lr:float = 1e-3, validation:bool = False, regularization:float = 0,
            train_dataloader = None, val_dataloader = None):
        print(f"Fitting and evaluating {self.name}...")
        self.to(self.device) # sending our model to desired device (GPU or CPU) for calculations
        if train_dataloader == None:
            train_dataloader = DataLoader(MyDataset(), batch_size=batch_size) # Setting dataloaders for efficiency
        train_size = len(train_dataloader)
        if validation:
            if val_dataloader:
                val_dataloader = DataLoader(MyDataset(val = True), batch_size=batch_size)
            val_size = len(val_dataloader)            
        # loading the optimizers with the networkd's parameters
        opt = Adam(list(self.parameters()), lr = lr, weight_decay=regularization)
        loss_fn = Processing.custom_loss(norm_coeff = self.norm_coeff, angle_coeff = self.angle_coeff,
                                         energy_coeff = self.energy_coeff, var_coeff = self.var_coeff,
                                         abs_coeff = self.abs_coeff, cross_coeff = self.cross_coeff)
        self.history["train_loss"] = []
        self.history["train_psnr"] = []
        if validation:
            self.history["val_loss"] = []
            self.history["val_psnr"] = []
        self.history["train_loss"] = []
        self.weights.append(copy.deepcopy(self.state_dict()))
        print("Training the network...")
        tic = time.perf_counter()
        for e in range(epochs): # beginning to train...
            tac = time.perf_counter()
            print("Epoch", e + 1, "/", epochs)
            self.train() # using Pytorch's method
            total_train_loss = 0
            total_train_psnr = 0
            total_val_loss = 0
            total_val_psnr = 0
            # evaluation
            if validation:
                print("Evaluating the model at current state...")
                with torch.no_grad(): # no need to backpropagate : saves time
                    self.eval()
                    for (image, label) in val_dataloader: # metrics calculations                        
                        with torch.cuda.amp.autocast(): # for efficiency purposes
                            pred = self(image)
                            # computing metrics
                            total_val_loss += loss_fn(pred, label).item()
                            total_val_psnr += Processing.PSNR(pred, label).item()
            for (image, label) in train_dataloader:
                pred = self(image)
                loss = loss_fn(pred, label)
                # backpropagating according to Pytorch's doc
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_train_loss += loss.item()
                total_train_psnr += Processing.PSNR(pred, label).item()           
            # saving this epoch's metrics
            self.history["train_loss"].append(total_train_loss / train_size)
            self.history["train_psnr"].append(total_train_psnr / train_size)
            if validation:
                # same for validation
                self.history["val_loss"].append(total_val_loss / val_size)
                self.history["val_psnr"].append(total_val_psnr / val_size)                
            # informating the user
            print(f"Time elapsed : {round(time.perf_counter() - tac)} s")
            print(f"Training loss : {round(total_train_loss / train_size, 4)}")
            print(f"Training PSNR : {round(total_train_psnr / train_size, 2)}")
            if validation:
                print(f"Validation loss : {round(total_val_loss / val_size, 2)}")
                print(f"Validation PSNR : {round(total_val_psnr / val_size, 2)} dB")
            self.weights.append(copy.deepcopy(self.state_dict()))
        print(f"Done in {round(time.perf_counter() - tic)} s.")
        # updating network state
        self.is_trained = True
        self.loss = total_train_loss
        self.PSNR = total_train_psnr
        self.plot_training_results() # to visually keep track of the network's progression
        plt.savefig("courbes.png")
        
    def save(self):
        if os.path.exists("networks/" + self.name + "/"):
            shutil.rmtree("networks/" + self.name + "/")
        os.mkdir("networks/" + self.name + "/")
        with open("networks/" + self.name + "/history.yaml", "w") as fichier:
            fichier.write(yaml.dump(self.history, Dumper = yaml.Dumper))
        torch.save(self.state_dict(), "networks/" + self.name + "/state_dict.yaml")
    
    def load(name):
        network = Network(name)
        network.load_state_dict(torch.load("networks/" + name + "/state_dict.yaml"))
        with open("networks/" + name + "/history.yaml", 'r') as fichier:
            network.history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
        return network

    def plot_training_results(self):
        history = self.history # loading history dict
        if history == None:
            raise Exception("No history, you have to fit the model first.")
        t = np.arange(len(history["train_loss"])) # x axis
        fix, ax = plt.subplots(2, figsize=(8,6))
        ax[0].set_title("Loss")
        ax[0].plot(t, history["train_loss"], label = "Training loss")
        if "val_loss" in history:
            ax[0].plot(t, history["val_loss"], label = "Validation loss")
        ax[0].legend()
        ax[1].set_title("PSNR (dB)")
        ax[1].plot(t, history["train_psnr"], label= "Training PSNR")
        if "val_psnr" in history:
            ax[1].plot(t, history["val_psnr"], label = "Validation PSNR")
        ax[1].legend()

    def plot_example_images(self, n_images = 1):
        fig, ax = plt.subplots(n_images, 3, figsize=(15, 5*n_images))
        ds = MyDataset(test = True)
        if n_images == 1:
            ax[0].set_title("Label")
            ax[1].set_title("Image")
            ax[2].set_title("Output")  
            pair = random.choice(ds)
            ax[0].imshow(np.squeeze(pair[1].cpu()), vmin = 0, vmax = 1)
            ax[1].imshow(np.squeeze(pair[0].cpu()), vmin = 0, vmax = 1)
            output = self(pair[0][None,...].detach().cpu()).detach().cpu()
            ax[2].imshow(np.squeeze(output), vmin = 0, vmax = 1)
        else:
            ax[0, 0].set_title("Label")
            ax[0, 1].set_title("Image")
            ax[0, 2].set_title("Output")
            for i in range(n_images):
                pair = random.choice(ds)
                ax[i, 0].imshow(np.squeeze(pair[1].cpu()), vmin = 0, vmax = 1)
                ax[i, 1].imshow(np.squeeze(pair[0].cpu()), vmin = 0, vmax = 1)
                ax[i, 2].imshow(np.squeeze(self(pair[0][None,...]).detach().cpu()), vmin = 0, vmax = 1)
        plt.show()    


class Deconv(Network):

    def __init__(self, name = "cnn", channels = 16, RL = False, mix = False, norm_coeff = 0, angle_coeff = 0,
                 energy_coeff = 1, var_coeff = 0, abs_coeff = 0, cross_coeff = 0):
        super(Deconv, self).__init__(name, norm_coeff, angle_coeff, energy_coeff, var_coeff, abs_coeff, cross_coeff)
        self.RL = RL
        self.mix = mix
        self.channels = channels
        rep = (channels//2, channels - channels//2 - channels//4, channels//4)
        # downsampler
        self.intro = Conv2d(in_channels=1, out_channels=channels, kernel_size=11, padding="same")
        self.layers.append(self.intro)
        self.layers.append(ReLU())
        self.layers.append([])
        self.layers[2] = [[], [], []]
        #first branch
        self.conv30 = Conv2d(in_channels=channels, out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv30)
        self.layers[2][0].append(ReLU())
        self.conv31 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv31)
        self.conv32 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv32)
        self.conv33 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv33)
        self.conv34 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv34)
        self.conv35 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv35)
        self.conv36 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv36)
        self.conv37 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv37)
        self.conv38 = Conv2d(in_channels=rep[0], out_channels=rep[0], kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv38)
        self.conv39 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same")
        self.layers[2][0].append(self.conv39)
        self.layers[2][0].append(ReLU())
        #second branch
        self.conv50 = Conv2d(in_channels=channels, out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv50)
        self.layers[2][1].append(ReLU())
        self.conv51 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv51)
        self.conv52 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv52)
        self.conv53 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv53)
        self.conv54 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv54)
        self.conv55 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv55)
        self.conv56 = Conv2d(in_channels=rep[1], out_channels=rep[1], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv56)
        self.conv57 = Conv2d(in_channels=rep[1] + rep[2], out_channels=rep[1] + rep[2], kernel_size=5, padding="same")
        self.layers[2][1].append(self.conv57)
        self.layers[2][1].append(ReLU())
        #third branch
        self.conv70 = Conv2d(in_channels=channels, out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv70)
        self.layers[2][2].append(ReLU())
        self.conv71 = Conv2d(in_channels=rep[2], out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv71)
        self.conv72 = Conv2d(in_channels=rep[2], out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv72)
        self.conv73 = Conv2d(in_channels=rep[2], out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv73)
        self.conv74 = Conv2d(in_channels=rep[2], out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv74)
        self.conv75 = Conv2d(in_channels=rep[2], out_channels=rep[2], kernel_size=7, padding="same")
        self.layers[2][2].append(self.conv75)
        self.layers[2][2].append(ReLU())
        #conclusion
        self.conc1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same")
        self.layers.append(self.conc1)
        self.conc2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same")
        self.layers.append(self.conc2)
        self.conc3 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same")
        self.layers.append(self.conc3)
        self.conc4 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding="same")
        self.layers.append(self.conc4)
        self.layers.append(ReLU())
        self.conc5 = Conv2d(in_channels=channels, out_channels=1, kernel_size=3, padding="same")
        self.layers.append(self.conc5)

    def forward(self, x:torch.tensor)->torch.tensor:
        input_ = x.detach().clone() # to avoid side effect
        for layer in self.layers[: 2]:
            x = layer(x)
        branch1, branch2, branch3 = self.layers[2][0], self.layers[2][1], self.layers[2][2]
        x1, x2, x3 = branch1[0](x), branch2[0](x), branch3[0](x)
        for layer in branch3[1: ]:
            x3 = layer(x3)
        for layer in branch2[1: -2]:
            x2 = layer(x2)
        x2 = torch.cat((x2, x3), 1)
        x2 = branch2[-2](x2)
        x2 = branch2[-1](x2)
        for layer in branch1[1: -2]:
            x1 = layer(x1)
        x1 = torch.cat((x1, x2), 1)
        x1 = branch1[-2](x1)
        x = branch1[-1](x1)
        for layer in self.layers[3: ]:
            x = layer(x)
        x = input_ + x # residual learning
        return x
    
    def save(self):
        if os.path.exists("networks/" + self.name + "/"):
            shutil.rmtree("networks/" + self.name + "/")
        self.history['channels'] = self.channels
        self.history['RL'] = self.RL
        self.history['mix'] = self.mix
        os.mkdir("networks/" + self.name + "/")
        with open("networks/" + self.name + "/history.yaml", "w") as fichier:
            fichier.write(yaml.dump(self.history, Dumper = yaml.Dumper))
        for i, state_dict in enumerate(self.weights):
            torch.save(self.weights[i], "networks/" + self.name + f"/state_dict{i}.yaml")
        torch.save(self.state_dict(), "networks/" + self.name + "/state_dict.yaml")
    
    def load(name, channels = 16, branch = True, epoch = None):
        if branch:
            with open("networks/" + name + "/history.yaml", 'r') as fichier:
                history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
        else:
            with open(name + "/history.yaml", 'r') as fichier:
                history = yaml.load(fichier, Loader = yaml.UnsafeLoader)
        if "channels" in history:
            channels = history["channels"]
        RL, mix = False, False
        if 'RL' in history:
            RL = history['RL']
        if 'mix' in history:
            mix = history['mix']
        network = Deconv(name, channels = channels, RL = RL, mix = mix, norm_coeff = history["norm_coeff"],
                         angle_coeff = history["angle_coeff"], energy_coeff = history["energy_coeff"],
                         var_coeff = history["var_coeff"], abs_coeff = history["abs_coeff"],
                         cross_coeff = history["cross_coeff"])
        if branch:
            if epoch == None:
                network.load_state_dict(torch.load("networks/" + name + "/state_dict.yaml", map_location=torch.device('cpu')))
            else:
                network.load_state_dict(torch.load("networks/" + name + f"/state_dict{epoch}.yaml", map_location=torch.device('cpu')))
        else:
            if epoch == None:
                network.load_state_dict(torch.load(name + "/state_dict.yaml", map_location=torch.device('cpu')))
            else:
                network.load_state_dict(torch.load(name + f"/state_dict{epoch}.yaml", map_location=torch.device('cpu')))
        network.history = history    
        return network

# +
#Generation.import_images(40, start_ind = 1)
#Generation.generate_dataset(sample_size = 300, n_patches = 3000)
#Processing.quality_grid(n_val = 50)
# -

