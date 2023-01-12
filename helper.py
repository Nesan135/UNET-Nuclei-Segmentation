import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2,os,math
import matplotlib.pyplot as plt

# helper functions

# Look for file in varying directories of main directory
def find(main_directory):
    for root, dirs, files in os.walk(os.getcwd()):
        if main_directory in files:
            return os.path.join(root, main_directory)

# change color of images to grayscale, resize to 128 x 128
def image_to_array(train_image_dir,train_masks_dir):
    image_list=[]
    masks_list=[]
    for ImageId in os.listdir(train_image_dir):
        img = cv2.imread(os.path.join(train_image_dir,ImageId))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(128,128))
        image_list.append(img)
        mask = cv2.imread(os.path.join(train_masks_dir,ImageId),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(128,128))
        masks_list.append(mask)
    return np.array(image_list), np.array(masks_list)

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask = tf.math.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]