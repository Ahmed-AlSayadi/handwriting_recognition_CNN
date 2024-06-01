#!/usr/bin/env python
# coding: utf-8

# In[4]:


# https://github.com/tuandoan998/Handwritten-Text-Recognition
# https://github.com/githubharald/SimpleHTR


# In[21]:


import keras

print(keras.__version__)


# ##### Enabling GPU for training the model 

# In[2]:


import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specify the index of the GPU device to use (e.g., 0, 1, 2, ...)


# In[4]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# ### Parameters of The System

# ####  The configuration parameters and hyperparameters used in the system. It defines variables such as image dimensions, batch size, learning rate, etc.

# In[12]:


letters = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_classes = len(letters) + 1

word_cfg = {
	'batch_size': 64,
	'input_length': 30,
	'model_name': 'iam_words',
	'max_text_len': 16,
	'img_w': 128,
	'img_h': 64
}

line_cfg = {
	'batch_size': 16,
	'input_length': 98,
	'model_name': 'iam_line',
	'max_text_len': 74,
	'img_w': 800,
	'img_h': 64
}


# ### Preprocessing Images

# #### The functions for preprocessing the input data, such as resizing images, converting them to grayscale, normalizing pixel values, and applying any required transformations.

# In[13]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def add_padding(img, old_w, old_h, new_w, new_h):
    h1, h2 = int((new_h-old_h)/2), int((new_h-old_h)/2)+old_h
    w1, w2 = int((new_w-old_w)/2), int((new_w-old_w)/2)+old_w
#######
    if len(img.shape) == 2:  # Grayscale image
        img_pad = np.ones([new_h, new_w]) * 255
        img_pad[h1:h2, w1:w2] = img
    else:  # Color image
        img_pad = np.ones([new_h, new_w, 3]) * 255
        img_pad[h1:h2, w1:w2, :] = img
    
    return img_pad
#####
'''
    img_pad = np.ones([new_h, new_w, 3]) * 255
    img_pad[h1:h2, w1:w2,:] = img
    return img_pad
'''
def fix_size(img, target_w, target_h):
    h, w = img.shape[:2]
    if w<target_w and h<target_h:
        img = add_padding(img, w, h, target_w, target_h)
    elif w>=target_w and h<target_h:
        new_w = target_w
        new_h = int(h*new_w/w)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    elif w<target_w and h>=target_h:
        new_h = target_h
        new_w = int(w*new_h/h)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    else:
        '''w>=target_w and h>=target_h '''
        ratio = max(w/target_w, h/target_h)
        new_w = max(min(target_w, int(w / ratio)), 1)
        new_h = max(min(target_h, int(h / ratio)), 1)
        new_img = cv.resize(img, (new_w, new_h), interpolation = cv.INTER_AREA)
        img = add_padding(new_img, new_w, new_h, target_w, target_h)
    return img

def preprocess(path, img_w, img_h):
    """ Pre-processing image for predicting """
    ### img = cv.imread(path)
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)  # Load the image as grayscale
    if img is None:
        raise ValueError(f"Failed to load image: {path}")
    img = fix_size(img, img_w, img_h)

    img = np.clip(img, 0, 255)
    img = np.uint8(img)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32)
    img /= 255
    print(path)
    return img

if __name__=='__main__':
    img = cv.imread('data/IAM/lines/a01/a01-000u/a01-000u-00.png', 0)
    plt.imshow(img, cmap='gray')
    plt.show()
    img = preprocess('data/IAM/lines/a01/a01-000u/a01-000u-00.png', 800, 64)
    print(img.shape)
    plt.imshow(img, cmap='gray')
    plt.show()


# ### Image Generatoor

# #### The data generator or data loading functions to efficiently load and augment the training data.

# In[14]:


import numpy as np
import random
from keras import backend as K
#from Preprocessor import preprocess
#from Parameter import *


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

class TextImageGenerator:
    
    def __init__(self, data,
                 img_w,
                 img_h, 
                 batch_size, 
                 i_len,
                 max_text_len):
        
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.samples = data
        self.n = len(self.samples)
        self.i_len = i_len
        self.indexes = list(range(self.n))
        self.cur_index = 0
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        for i, (img_filepath, text) in enumerate(self.samples):
            img = preprocess(img_filepath, self.img_w, self.img_h)
            self.imgs[i, :, :] = img
            self.texts.append(text)
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.zeros([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * self.i_len
            label_length = np.zeros((self.batch_size, 1))
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i, :len(text)] = text_to_labels(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
            }
            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


# ### CRNN (Convolutional Recurrent Neural Network) model

# #### The implementation of the CRNN (Convolutional Recurrent Neural Network) model architecture.

# In[11]:


#from Parameter import *
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from keras.models import Model
from keras.layers import GRU
#from keras.layers.merge import add, concatenate
from keras.layers import add, concatenate
####
#from tensorflow.keras.layers import GRU
#from tensorflow.keras.layers import add, concatenate
####

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def word_model():
    img_w = word_cfg['img_w']
    img_h = word_cfg['img_h']
    max_text_len = word_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    # Make Networkw
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
    gru1_merged = BatchNormalization()(gru1_merged)
    
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
    gru2_merged = BatchNormalization()(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(gru2_merged) #(None, 32, 80)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=input_data, outputs=y_pred)
    model_predict.summary()

    return model, model_predict


def line_model():
    img_w = line_cfg['img_w']
    img_h = line_cfg['img_h']
    max_text_len = line_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    # Make Networkw
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 800, 64, 1)

    # Convolution layer
    inner = Conv2D(64, (5, 5), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  # (None, 800, 64, 64)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,400, 32, 64)

    inner = Conv2D(128, (5, 5), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 400, 32, 128)
    inner = Activation('relu')(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 400, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 200, 16, 128)
    
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 256)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(256, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 256)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(512, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    
    inner = Conv2D(512, (3, 3), padding='same', name='conv7', kernel_initializer='he_normal')(inner)  # (None, 200, 16, 512)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max3')(inner)  # (None, 100, 8, 512)

    # CNN to RNN
    inner = Reshape(target_shape=((100, 4096)), name='reshape')(inner)  # (None, 100, 4096)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 100, 64)

    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 100, 512)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 100, 512)
    gru1_merged = BatchNormalization()(gru1_merged)
    
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 100, 1024)
    gru2_merged = BatchNormalization()(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(80, kernel_initializer='he_normal',name='dense2')(gru2_merged) #(None, 100, 80)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=input_data, outputs=y_pred)
    model_predict.summary()

    # Convert model to JSON
    model_json = model.to_json()
    model_predict_json = model_predict.to_json()

    # Save JSON to file
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    
    with open('model_prediction.json', 'w') as json_file:
        json_file.write(model_predict_json)

    
    return model, model_predict


# In[3]:


#from Parameter import *
from keras import backend as K
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Reshape, Lambda, BatchNormalization
from keras.models import Model
#from keras.layers.recurrent import GRU
#from keras.layers.merge import add, concatenate

####
from keras.layers import GRU
from keras.layers import add, concatenate
#from tensorflow.keras.layers import add, concatenate
####

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def word_model():
    img_w = word_cfg['img_w']
    img_h = word_cfg['img_h']
    max_text_len = word_cfg['max_text_len']
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_w, img_h)
    else:
        input_shape = (img_w, img_h, 1)

    # Make Networkw
    input_data = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)

    # Convolution layer (VGG)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(input_data)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to RNN
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
    gru_1 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 32, 512)
    gru_1b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    reversed_gru_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_1b)

    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 32, 512)
    gru1_merged = BatchNormalization()(gru1_merged)
    
    gru_2 = GRU(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)
    reversed_gru_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (gru_2b)

    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 32, 1024)
    gru2_merged = BatchNormalization()(gru2_merged)

    # transforms RNN output to character activations:
    inner = Dense(num_classes, kernel_initializer='he_normal',name='dense2')(gru2_merged) #(None, 32, 80)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # loss function
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length]
    )

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model_predict = Model(inputs=input_data, outputs=y_pred)
    model_predict.summary()


    # Convert model to JSON
    model_json = model.to_json()
    model_predict_json = model_predict.to_json()
    
    # Save JSON to file
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    with open('model_prediction.json', 'w') as json_file:
        json_file.write(model_predict_json)
    print("JSON files created successfully.")
    return model, model_predict


# In[ ]:





# In[6]:


'''import json
from keras.models import model_from_json
model, model_predict = word_model()
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Load the JSON model structure from the file
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create a new model from the JSON structure
loaded_model = model_from_json(loaded_model_json)'''


# In[ ]:





# ### Utility Functions

# #### The functions for calculating evaluation metrics, loading/saving models, visualizing results, etc

# In[7]:


import os
import numpy as np
import itertools
#from Parameter import *
#from Preprocessor import preprocess
from keras import backend as K


def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], 1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for c in out_best:
        if c < len(letters):
            outstr += letters[c]
    return outstr

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret

def get_paths_and_texts(partition_split_file, is_words):
    paths_and_texts = []
    
    with open (partition_split_file) as f:
            partition_folder = f.readlines()
    partition_folder = [x.strip() for x in partition_folder]
    
    if is_words:
        with open ('data/IAM/words.txt') as f:
            for line in f:
                if not line or line.startswith('#'): # comment in txt file
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err': # er: segmentation of word can be bad
                    continue

                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join('data/IAM/words', label_dir, sub_label_dir, fn)

                gt_text = ' '.join(line_split[8:])
                if len(gt_text)>16:
                    continue

                if sub_label_dir in partition_folder:
                    paths_and_texts.append([img_path, gt_text])
        
    else:
        with open('data/IAM/lines.txt') as f:
            for line in f:
                if not line or line.startswith('#'):
                    continue
                line_split = line.strip().split(' ')
                assert len(line_split) >= 9
                status = line_split[1]
                if status == 'err':
                    continue
                file_name_split = line_split[0].split('-')
                label_dir = file_name_split[0]
                sub_label_dir = '{}-{}'.format(file_name_split[0], file_name_split[1])
                fn = '{}.png'.format(line_split[0])
                img_path = os.path.join('data/IAM/lines', label_dir, sub_label_dir, fn)
                gt_text = ' '.join(line_split[8:])
                gt_text = gt_text.replace('|', ' ')
                l = len(gt_text)
                if l<10 or l>74:
                    continue
                paths_and_texts.append([img_path, gt_text])
    return paths_and_texts

def predict_image(model_predict, path, is_word):
    if is_word:
        width = word_cfg['img_w']
    else:
        width = line_cfg['img_w']
    img = preprocess(path, width, 64)
    img = img.T
    if K.image_data_format() == 'channels_first':
        img = np.expand_dims(img, 0)
    else:
        img = np.expand_dims(img, -1)
    img = np.expand_dims(img, 0)

    net_out_value = model_predict.predict(img)
    pred_texts = decode_label(net_out_value)
    return pred_texts


# ### Words Segmentation

# #### This contain functions for segmenting the input image into individual words or lines.

# In[8]:


import math
import cv2
import numpy as np


def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
	"""Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf
	
	Args:
		img: grayscale uint8 image of the text-line to be segmented.
		kernelSize: size of filter kernel, must be an odd integer.
		sigma: standard deviation of Gaussian function used for filter kernel.
		theta: approximated width/height ratio of words, filter function is distorted by this factor.
		minArea: ignore word candidates smaller than specified area.
		
	Returns:
		List of tuples. Each tuple contains the bounding box and the image of the segmented word.
	"""

	# apply filter kernel
	kernel = createKernel(kernelSize, sigma, theta)
	imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
	(_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	imgThres = 255 - imgThres

	# find connected components. OpenCV: return type differs between OpenCV2 and 3
	if cv2.__version__.startswith('3.'):
		(_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	else:
		(components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# append components to result
	res = []
	for c in components:
		# skip small word candidates
		if cv2.contourArea(c) < minArea:
			continue
		# append bounding box and image of word to result list
		currBox = cv2.boundingRect(c) # returns (x, y, w, h)
		(x, y, w, h) = currBox
		currImg = img[y:y+h, x:x+w]
		res.append((currBox, currImg))

	# return list of words, sorted by x-coordinate
	return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
	"""convert given image to grayscale image (if needed) and resize to desired height"""
	assert img.ndim in (2, 3)
	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	h = img.shape[0]
	factor = height / h
	return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
	"""create anisotropic filter kernel according to given parameters"""
	assert kernelSize % 2 # must be odd size
	halfSize = kernelSize // 2
	
	kernel = np.zeros([kernelSize, kernelSize])
	sigmaX = sigma
	sigmaY = sigma * theta
	
	for i in range(kernelSize):
		for j in range(kernelSize):
			x = i - halfSize
			y = j - halfSize
			
			expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
			xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
			yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)
			
			kernel[i, j] = (xTerm + yTerm) * expTerm

	kernel = kernel / np.sum(kernel)
	return kernel


# ### Training The model

# #### The functions for initializing the model, defining the loss function and optimizer, and running the training loop

# In[13]:


#from Parameter import *
#from ImageGenerator import TextImageGenerator
#from CRNN_Model import word_model, line_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
#from Utils import *


def train(train_data, val_data, is_word_model):
    if is_word_model:
        model, _ = word_model()
        cfg = word_cfg
    else:
        model, _ = line_model()
        cfg = line_cfg

    input_length = cfg['input_length']
    model_name = cfg['model_name']
    max_text_len = cfg['max_text_len']
    img_w = cfg['img_w']
    img_h = cfg['img_h']
    batch_size = cfg['batch_size']
    train_set = TextImageGenerator(train_data, img_w, img_h, batch_size, input_length, max_text_len)
    print('Loading data for train ...')
    train_set.build_data()
    val_set = TextImageGenerator(val_data, img_w, img_h, batch_size, input_length, max_text_len)
    val_set.build_data()
    print('Done')
    
    print("Number train samples: ", train_set.n)
    print("Number val samples: ", val_set.n)
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    ckp = ModelCheckpoint(
        filepath='Resource/'+model_name+'--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set.next_batch(),
                        steps_per_epoch=train_set.n // batch_size,
                        epochs=32,
                        validation_data=val_set.next_batch(),
                        validation_steps=val_set.n // batch_size,
                        callbacks=[ckp, earlystop])

    return model


# In[14]:


if __name__=='__main__':
    train_data = get_paths_and_texts('data/IAM/splits/train.uttlist', is_words=True)
    val_data = get_paths_and_texts('data/IAM/splits/validation.uttlist', is_words=True)
    print('number of train image: ', len(train_data))
    print('number of valid image: ', len(val_data))

    model = train(train_data, val_data, True)


# In[ ]:





# In[10]:


'''

#from Parameter import *
#from ImageGenerator import TextImageGenerator
#from CRNN_Model import word_model, line_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
#from Utils import *


def train(train_data, val_data, is_word_model):
    if is_word_model:
        model, _ = word_model()
        cfg = word_cfg
    else:
        model, _ = line_model()
        cfg = line_cfg

    input_length = cfg['input_length']
    model_name = cfg['model_name']
    max_text_len = cfg['max_text_len']
    img_w = cfg['img_w']
    img_h = cfg['img_h']
    batch_size = cfg['batch_size']
    train_set = TextImageGenerator(train_data, img_w, img_h, batch_size, input_length, max_text_len)
    print('Loading data for train ...')
    train_set.build_data()
    val_set = TextImageGenerator(val_data, img_w, img_h, batch_size, input_length, max_text_len)
    val_set.build_data()
    print('Done')
    
    print("Number train samples: ", train_set.n)
    print("Number val samples: ", val_set.n)
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')

    ckp = ModelCheckpoint(
        filepath='Resource/'+model_name+'--{epoch:02d}--{val_loss:.3f}.h5', monitor='val_loss',
        verbose=1, save_best_only=True, save_weights_only=True
    )
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'
    )

    model.fit_generator(generator=train_set.next_batch(),
                        steps_per_epoch=train_set.n // batch_size,
                        epochs=32,
                        validation_data=val_set.next_batch(),
                        validation_steps=val_set.n // batch_size,
                        callbacks=[ckp, earlystop])

    return model

if __name__=='__main__':
    train_data = get_paths_and_texts('data/IAM/splits/train.uttlist', is_words=True)
    val_data = get_paths_and_texts('data/IAM/splits/validation.uttlist', is_words=True)
    print('number of train image: ', len(train_data))
    print('number of valid image: ', len(val_data))
#'''
'''
# ...

    if __name__=='__main__':
        train_data = get_paths_and_texts('data/IAM/splits/train.uttlist', is_words=True)
        print('Number of train images:', len(train_data))
    
        # Print a sample of train_data
        num_samples = 5  # Number of samples to print
        sample_data = train_data[:num_samples]  # Select the first num_samples
    
        for image_path, label in sample_data:
            print('Image path:', image_path)
            print('Label:', label)
            print('---')

        # Print and display a sample of train_data
        num_samples = 5  # Number of samples to display
        sample_data = train_data[:num_samples]  # Select the first num_samples
    
        for image_path, label in sample_data:
            # Load and display the image
            img = Image.open(image_path).convert('L')
            plt.imshow(img, cmap='gray')
            plt.title('Image')
            plt.axis('off')
            plt.show()
    
            # Print the corresponding label
            #print('Label:', label)
            #print('---')
        

    model = train(train_data, val_data, True)


'''


# ### Evaluating Word Model

# #### Evaluating the trained model on word-level

# In[15]:


from sklearn.model_selection import train_test_split
import editdistance
from keras.models import model_from_json 
import tensorflow as tf
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


'''
from Spell import correction
from ImageGenerator import TextImageGenerator
from Parameter import *
from Utils import *
'''

if __name__=='__main__':
	test_data = get_paths_and_texts('data/IAM/splits/test.uttlist', is_words=True)
	print('number of test image: ', len(test_data))

	with open('Resource/model_prediction.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights('Resource/iam_words--17--1.887.h5')

	ed_chars = num_chars = ed_words = num_words = 0
	for path, gt_text in test_data:
		pred_text = predict_image(model, path, is_word=True)
		if gt_text!=pred_text:
			ed_words += 1 
		num_words += 1
		ed_chars += editdistance.eval(gt_text, pred_text)
		num_chars += len(gt_text)
	# batch_size = word_cfg['batch_size']
	# test_set = TextImageGenerator(test_data, word_cfg['img_w'], word_cfg['img_h'], batch_size, word_cfg['input_length'], word_cfg['max_text_len'])
	# print('Loading data for evaluation ...')
	# test_set.build_data()
	# print('Done')
	# print("Number test set: ", test_set.n)

	# batch = 0
	# num_batch = int(test_set.n/batch_size)
	# for inp_value, _ in test_set.next_batch():
	# 	if batch>=num_batch:
	# 		break
	# 	print('batch: %s/%s' % (batch, str(num_batch)))

	# 	labels = inp_value['the_labels']
	# 	label_len = inp_value['label_length']
	# 	g_texts = []
	# 	for label in labels:
	# 		g_text = ''.join(list(map(lambda x: letters[int(x)], label)))
	# 		g_texts.append(g_text)
	# 	pred_texts = decode_batch(model.predict(inp_value))

	# 	for i in range(batch_size):
	# 		g_texts[i] = g_texts[i][:int(inp_value['label_length'].item(i))]
	# 		ed_chars += editdistance.eval(g_texts[i], pred_texts[i])
	# 		num_chars += len(g_texts[i])
	# 		if g_texts[i]!=pred_texts[i]:
	# 			ed_words += 1 
	# 		num_words += 1
 			# batch += 1

		print('ED chars: ', ed_chars)
		print('ED words: ', ed_words)

	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)


# ### Evaluating Line Model

# #### Evaluating the trained model on line-level

# In[23]:


'''

from keras.models import model_from_json
from sklearn.model_selection import train_test_split
import editdistance
import os
import cv2
import numpy as np
from keras import backend as K
import shutil
import re
'''
'''
from Utils import *
from WordSegmentation import wordSegmentation, prepareImg
from Preprocessor import preprocess
from Utils import get_paths_and_texts
from Spell import correction_list
from ImageGenerator import TextImageGenerator
'''
'''
pattern = '[' + r'\w' + ']+'
def getWordIDStrings(s1, s2):
	# get words in string 1 and string 2
	words1 = re.findall(pattern, s1)
	words2 = re.findall(pattern, s2)
	# find unique words
	allWords = list(set(words1 + words2))
	# list of word ids for string 1
	idStr1 = []
	for w in words1:
		idStr1.append(allWords.index(w))
	# list of word ids for string 2
	idStr2 = []
	for w in words2:
		idStr2.append(allWords.index(w))
	return (idStr1, idStr2)

def detect_word_model(model_predict, test_img):
	img = prepareImg(cv2.imread(test_img), 64)
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')
	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		cv2.imwrite('tmp/%d.png'%j, wordImg)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	pred_line = []
	for f in imgFiles:
		pred_line.append(predict_image(model_predict, 'tmp/'+f, True))
	shutil.rmtree('tmp')
	pred_line = correction_list(pred_line)
	return (' '.join(pred_line))
####
#paths_and_texts = get_paths_and_texts('data/IAM/lines.txt', is_words=False)
####
if __name__=='__main__':
	paths_and_texts = get_paths_and_texts(is_words=False)
	print('number of image: ', len(paths_and_texts))

	paths_and_texts_train, paths_and_texts_test = train_test_split(paths_and_texts, test_size=0.3, random_state=1707)
	paths_and_texts_val, paths_and_texts_test = train_test_split(paths_and_texts_test, test_size=0.65, random_state=1707)
	print('number of train image: ', len(paths_and_texts_train))
	print('number of valid image: ', len(paths_and_texts_val))
	print('number of test image: ', len(paths_and_texts_test))

	#with open('Resource/word_model_predict.json', 'r') as f:
	#	model_predict = model_from_json(f.read())
	#model_predict.load_weights('Resource/iam_words--15--1.791.h5')
	with open('Resource/line_model_predict.json', 'r') as f:
		model_predict = model_from_json(f.read())
	model_predict.load_weights('Resource/iam_lines--12--17.373.h5')

	batch_size = line_cfg['batch_size']
	test_set = TextImageGenerator(paths_and_texts_test, line_cfg['img_w'], line_cfg['img_h'], batch_size, line_cfg['input_length'], line_cfg['max_text_len'])
	print('Loading data for evaluation ...')
	test_set.build_data()
	print('Done')
	print("Number test set: ", test_set.n)

	ed_chars = num_chars = ed_words = num_words = 0
	batch = 0
	num_batch = int(test_set.n/batch_size)
	for inp_value, _ in test_set.next_batch():
		if batch>=num_batch:
			break
		print('batch: %s/%s' % (batch, str(num_batch)))

		labels = inp_value['the_labels']
		label_len = inp_value['label_length']
		g_texts = []
		for label in labels:
			g_text = ''.join(list(map(lambda x: letters[int(x)], label)))
			g_texts.append(g_text)
		pred_texts = decode_batch(model_predict.predict(inp_value))

		for i in range(batch_size):
			g_texts[i] = g_texts[i][:int(inp_value['label_length'].item(i))]
			ed_chars += editdistance.eval(g_texts[i], pred_texts[i])
			num_chars += len(g_texts[i])
			(idStrGt, idStrPred) = getWordIDStrings(g_texts[i], pred_texts[i])
			ed_words += editdistance.eval(idStrGt, idStrPred)
			num_words += len(idStrGt)

		print('ED chars: ', ed_chars)
		print('ED words: ', ed_words)
		batch += 1
	print('CER: ', ed_chars / num_chars)
	print('WER: ', ed_words / num_words)

'''


# ### Spell Checking

# #### The functions for spell checking or post-processing the recognized text to improve its correctness.

# In[16]:


import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

words_count = Counter(words(open('Resource/big.txt').read()))
checked_word = words(open('Resource/wordlist_mono_clean.txt').read())

def P(word, N=sum(words_count.values())): 
    "Probability of `word`."
    return words_count[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    if word.lower() in checked_word:
        new_word = word
    else:
        new_word = max(candidates(word, words_count), key=P)
        if word[0].lower()==new_word[0]:
            new_word = list(new_word)
            new_word[0]=word[0]
            new_word = ''.join(new_word)
    return new_word

def correction_list(words):
    res = []
    for word in words:
        if word.lower() in checked_word:
            new_word = word
        else:
            new_word = max(candidates(word), key=P)
            if word[0].lower()==new_word[0]:
                new_word = list(new_word)
                new_word[0]=word[0]
                new_word = ''.join(new_word)
        res.append(new_word)
    return res

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of words_count."
    return set(w for w in words if w in words_count)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

if __name__=='__main__':
    print(correction('Smell'))


# ### Predicting Handwritten Images

# #### The functions to perform predictions on new, unseen data using the trained model

# In[20]:


import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from keras.models import model_from_json
import shutil
from keras import backend as K
from keras.utils import plot_model
'''
from Utils import *
from WordSegmentation import wordSegmentation, prepareImg
from Preprocessor import preprocess
from Spell import correction_list
'''


if __name__=='__main__':
	#l_model, l_model_predict = line_model()
	#with open('line_model_predict.json', 'w') as f:
	#	f.write(l_model_predict.to_json())
	#with open('Resource/line_model_predict.json', 'r') as f:
		#l_model_predict = model_from_json(f.read())
	with open('Resource/model_prediction.json', 'r') as f:
		w_model_predict = model_from_json(f.read())
	#plot_model(l_model_predict, to_file='line_model.png', show_shapes=True, show_layer_names=True)
	w_model_predict.load_weights('Resource/iam_words--17--1.887.h5')
	#l_model_predict.load_weights('Resource/iam_lines--12--17.373.h5')
	test_img = 'Resource/test/6.png'
	
	img = prepareImg(cv2.imread(test_img), 64)
	img2 = img.copy()
	res = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=100)
	if not os.path.exists('tmp'):
		os.mkdir('tmp')

	for (j, w) in enumerate(res):
		(wordBox, wordImg) = w
		(x, y, w, h) = wordBox
		cv2.imwrite('tmp/%d.png'%j, wordImg)
		cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),1) # draw bounding box in summary image

	cv2.imwrite('Resource/test/summary.png', img2)
	plt.imshow(img2)
	imgFiles = os.listdir('tmp')
	imgFiles = sorted(imgFiles)
	pred_line = []
	for f in imgFiles:
		pred_line.append(predict_image(w_model_predict, 'tmp/'+f, True))
	print('-----------PREDICT-------------')
	print('[Word model]: '+' '.join(pred_line))
	pred_line = correction_list(pred_line)
	print('[Word model with spell]: '+' '.join(pred_line))
	
	#print('[Line model]: ' + predict_image(l_model_predict, test_img, False))

	plt.show()
	shutil.rmtree('tmp')


# In[ ]:


# Norvig, P. (2016, August). How to write a spelling corrector. Norvig. Retrieved February 28, 2024, from http://norvig.com/spell-correct.html


# In[ ]:




