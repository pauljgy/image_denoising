import os
import time
import numpy as np

from torch import Tensor
from torch import nn
from PIL import Image
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr

def listdir(path):
    return [f for f in os.listdir(path) if not f.startswith('.')]

def detach(tensor):
    return Tensor.cpu(tensor).detach().numpy() 

def create_training_data(filepath: str, target_array: np.ndarray):

    i = 0 # image number

    for folder in listdir(filepath):
        for image in listdir(os.path.join(filepath, folder)):
            if (i % 1000 == 0 ):
                print(f"image {i}")
            open_image = Image.open(os.path.join(filepath, folder, image))
            target_array[i][0] = open_image
            i += 1

    return None

def avg_psnr(test_size: int, images_true, images_test):

    total = 0

    for i in range(test_size):
        test_image = detach(images_test[i][0])
        true_image = detach(images_true[i][0])

        d_range = np.max([np.max(test_image) - np.min(test_image), np.max(true_image) - np.min(true_image)])
        temp = psnr(true_image, test_image, data_range=d_range)
        if (np.isnan(temp)):
            temp = 0
        total += temp
    
    return total/test_size

def show_images(image1: np.ndarray, image2: np.ndarray):
    f, plots = plt.subplots(1, 2)
    plots[0].imshow(image1, cmap="gray")
    plots[1].imshow(image2, cmap="gray")
    plt.show()

    return None

def show_images_hist(image1: np.ndarray, image2: np.ndarray):
    g, plots = plt.subplots(1, 2)
    plots[0].hist(image1)
    plots[0].set_title("Pixel dist. for im1")

    plots[1].hist(image2)
    plots[1].set_title("Pixel dist. for im2")
    plt.show()

    return None

def train(x: Tensor, y: Tensor, neural_network: nn.Module, epochs: int, batch_size: int, learning_rate: float, loss_function, optimizer):

    print("Starting training...")
    start_time = time.time()
    loss_log = np.zeros(epochs)
    num_batches = len(x)/batch_size

    for num in range(epochs):
        
        i = 0
        epoch_loss = 0

        while (i + batch_size < len(x)):
            samples = x[i:i+batch_size]
            labels = y[i:i+batch_size]
            y_hat = neural_network(samples)

            optimizer.zero_grad()
            loss = loss_function(labels, y_hat)
            if (i % 8000 == 0):
                print(loss)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            i += batch_size

        print(f"Epoch {num} complete. Elapsed time: {round(time.time() - start_time)}s")
        #print(f"Average loss: {epoch_loss/num_batches}")
        
        loss_log[num] = epoch_loss/num_batches
    
    print("Done training.")
    return loss_log
