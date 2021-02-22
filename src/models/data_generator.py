import os
import cv2 as cv
import numpy as np
from numpy import expand_dims

#Keras Libraries
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

#Libraries to plot
import matplotlib.pyplot as plt
import io
from io import BytesIO
from PIL import Image
from google.colab import files

from google.colab import drive  
drive.mount('/content/drive')   

# Use the Image Data Generator to import the images from the dataset
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   rotation_range = 100,
                                   width_shift_range = 0.5,
                                   height_shift_range = 0.5,                               
                                   vertical_flip = True,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Baggage_Screening/train/',
                                                 target_size = (400, 400),
                                                 batch_size = 24,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Baggage_Screening/test',
                                            target_size = (400, 400),
                                            batch_size = 24,
                                            class_mode = 'categorical')
     