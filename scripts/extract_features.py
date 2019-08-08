"""
this file extracts the features from various images. It would be executed
separately for training images, validation images and testing images.
"""

import argparse
import os
import h5py
import numpy as np
import tensorflow as tf
import cv2
import tensorflow.keras.applications.resnet50 as resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.backend import eval

parser = argparse.ArgumentParser()
parser.add_argument('--extract_features_for', default='train', type=str,
                    help='Please provide in the type of dataset for which you'
                         'want to extract features. The allowed dataset are'
                         'train, val and test.')
parser.add_argument('--input_image_dir', default=None)
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_file', default='../output/train_features.h5',
                    type=str)

parser.add_argument('--image_height', default=240, type=int)
parser.add_argument('--image_width', default=240, type=int)

parser.add_argument('--model', default='ResNet50', type=str)
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=1, type=int)


def build_model(args):
    if not hasattr(resnet50, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if 'resnet' not in args.model.lower():
        raise ValueError('Feature extraction only supports ResNets')

    '''
    Transfer learning: it helps to utilise previously trained network, rather
    than training new complex models by using their learned weights and then use
    standard training methods to learn the remaining, non-reused parameters.
    '''
    original_model = getattr(tf.keras.applications.resnet50, 'ResNet50')(
        weights='imagenet')

    bottleneck_input = original_model.get_layer(index=0).input
    bottleneck_output = original_model.get_layer(index=-2).output
    bottleneck_model = Model(inputs=bottleneck_input, outputs=bottleneck_output)

    '''Set every layer in the model to be non-trainable'''
    for layer in bottleneck_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(bottleneck_model)
    '''This output of the model is flattened  into (2048, ).'''

    '''Check the link 
    https://towardsdatascience.com/how-to-train-your-model-dramatically-faster-9ad063f0f718
    for fitting
    '''
    return model


def run_batch(current_batch, model):
    image_batch = np.concatenate(current_batch, 0).astype(np.float32)

    '''Normalise the input RGB image batch'''
    image_batch = tf.keras.utils.normalize(image_batch)
    image_batch = tf.convert_to_tensor(image_batch)

    '''Pass the input batch to the model to obtain the features'''
    features = model(image_batch)

    '''
    eval method helps in returning the actual value contained within the tensor
    '''
    return eval(features)


def main(args):
    input_paths = []

    '''Creates a set with no duplicate elements'''
    idx_set = set()

    for file in os.listdir(args.input_image_dir):
        if not file.endswith('.png'):
            continue

        '''Fetch the image number from the image'''
        idx = int(os.path.splitext(file)[0].split('_')[-1])

        '''Append a tuple to the list of the form  (filename, fileindex)'''
        input_paths.append((os.path.join(args.input_image_dir, file), idx))
        idx_set.add(idx)

    '''sort the names in the array in ascending order of the indexes'''
    input_paths.sort(key=lambda x: x[1])

    '''assert the consistency of set with the number of image files we have'''
    assert len(idx_set) == len(input_paths)
    assert min(idx_set) == 0 and max(idx_set) == len(input_paths) - 1

    if args.max_images is not None:
        input_paths = input_paths[:args.max_images]

    model = build_model(args)

    '''Create a h5py dataset that stores the feature of images'''
    with h5py.File(args.output_h5_file, 'w') as f:
        features_dataset = None
        i0 = 0
        current_batch = []
        for i, (path, idx) in enumerate(input_paths):
            img = cv2.imread(path)
            img = img[None]
            current_batch.append(img)
            if len(current_batch) == args.batch_size:
                features = run_batch(current_batch, model)
                if features_dataset is None:
                    N = len(input_paths)
                    _, C = features.shape
                    features_dataset = f.create_dataset('features', (N, C),
                                                        dtype=np.float32)
                i1 = i0 + len(current_batch)
                features_dataset[i0:i1] = features
                i0 = i1
                print('Processed %d / %d images' % (i1, len(input_paths)))
                current_batch.clear()

        if len(current_batch) > 0:
            features = run_batch(current_batch, model)
            i1 = i0 + len(current_batch)
            features_dataset[i0:i1] = features
            print('Processed %d / %d images' % (i1, len(input_paths)))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.extract_features_for == 'train' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/train/images/'
        args.output_h5_file = '../output/features/train_features.h5'
    elif args.extract_features_for == 'val' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/val/images/'
        args.output_h5_file = '../output/features/val_features.h5'
    elif args.extract_features_for == 'test' and args.input_image_dir is None:
        args.input_image_dir = '../../clevr-dataset-gen/output/test/images/'
        args.output_h5_file = '../output/features/test_features.h5'

    main(args)
