'''
Train autoencoder

Program for training a Convolutional Autoencoder.

Oliver Edholm, 14 years old 2016-01-03 11:24
'''
# imports
import os
import re
import logging
import numpy as np

import tflearn
from tflearn.data_utils import image_preloader
from tflearn.layers.core import input_data, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing

# setup
logging.basicConfig(level=logging.DEBUG)

# variables
TRAINING_DATA_PATH = 'images'

CHECKPOINTS_DIR_PATH = 'checkpoints'
CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR_PATH, 'conv_autoencoder_weights')

IMAGE_INPUT_SIZE = (256, 256)
BATCH_SIZE = 16


# functions
def load_data():
    logging.info('preparing data, can take a while')
    return image_preloader(TRAINING_DATA_PATH, image_shape=IMAGE_INPUT_SIZE,
                           mode='folder', filter_channel=True,
                           normalize=True)


def build_model():
    logging.info('building model')
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

    encoder = input_data(shape=(None, IMAGE_INPUT_SIZE[0], IMAGE_INPUT_SIZE[1],
                                3), data_preprocessing=img_prep)
    encoder = conv_2d(encoder, 16, 7, activation='relu')
    encoder = dropout(encoder, 0.25)  # you can have noisy input instead
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 16, 7, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    encoder = conv_2d(encoder, 8, 7, activation='relu')
    encoder = max_pool_2d(encoder, 2)
    
    decoder = conv_2d(encoder, 8, 7, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 16, 7, activation='relu')
    decoder = upsample_2d(decoder, 2)
    decoder = conv_2d(decoder, 3, 7)

    encoded_str = re.search(r', (.*)\)', str(encoder.get_shape)).group(1)
    encoded_size = np.prod([int(o) for o in encoded_str.split(', ')])
    
    original_img_size = np.prod(IMAGE_INPUT_SIZE) * 3
    
    percentage = round(encoded_size / original_img_size, 2) * 100
    logging.debug('the encoded representation is {}% of the original \
image'.format(percentage))
    
    return regression(decoder, optimizer='adadelta',
                      loss='binary_crossentropy', learning_rate=0.005)


def main():
    X, _ = load_data()

    conv_autencoder = build_model()

    logging.info('training')
    model = tflearn.DNN(conv_autencoder, tensorboard_verbose=3)
    model.fit(X, X, n_epoch=20, shuffle=True, show_metric=True,
              batch_size=BATCH_SIZE, validation_set=0.1, snapshot_epoch=True,
              run_id='selfie_conv_autoencoder',
              checkpoint_path=CHECKPOINTS_PATH)


if __name__ == '__main__':
    main()

