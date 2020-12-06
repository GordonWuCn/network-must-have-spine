import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt


def readData(path, num_classes=24, maxEty=10000, test_ratio=.3):
    id2label = {}
    images = []
    labels = []
    for idx, file in enumerate(os.listdir(path)):
        label = file.split('_')[-1][:-4]
        id2label[idx] = label

        images.append(np.load(path + '/' + file)[:maxEty])
        labels.extend([idx] * maxEty)

    images = tf.concat(images, axis=0)
    labels = tf.convert_to_tensor(labels)

    indices = tf.range(start=0, limit=tf.shape(images)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    test_size = int(shuffled_indices.shape[0] * test_ratio)
    train_idx, test_idx = shuffled_indices[test_size:], shuffled_indices[:test_size]

    train_inputs = tf.gather(images, train_idx)
    train_labels = tf.gather(labels, train_idx)
    test_inputs = tf.gather(images, test_idx)
    test_labels = tf.gather(labels, test_idx)

    train_inputs = tf.reshape(train_inputs, [train_inputs.shape[0], 28, 28, 1])
    test_inputs = tf.reshape(test_inputs, [test_inputs.shape[0], 28, 28, 1])
    train_labels = tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)

    return train_inputs, train_labels, test_inputs, test_labels, id2label


def scratch_data(path, num_classes=24, maxEty=10000, test_ratio=.3):
    train_imgs, train_labels, test_imgs, test_labels, id2label = readData(path, num_classes, maxEty, test_ratio)
    train_imgs = tf.cast(train_imgs, tf.float32) / 256
    test_imgs = tf.cast(test_imgs, tf.float32) / 256

    return train_imgs, train_labels, test_imgs, test_labels, id2label


def transfer_data(path, num_classes=24, maxEty=10000, test_ratio=.3):
    train_imgs, train_labels, test_imgs, test_labels, id2label = readData(path, num_classes, maxEty, test_ratio)

    train_imgs = tf.image.resize(train_imgs, [224, 224])
    train_imgs = tf.image.grayscale_to_rgb(train_imgs)
    train_imgs = tf.keras.applications.vgg19.preprocess_input(train_imgs)

    test_imgs = tf.image.resize(test_imgs, [224, 224])
    test_imgs = tf.image.grayscale_to_rgb(test_imgs)
    test_imgs = tf.keras.applications.vgg19.preprocess_input(test_imgs)

    return train_imgs, train_labels, test_imgs, test_labels, id2label


def transfer_data_large(path, num_classes=24, maxEty=10000, test_ratio=.3):
    def transform(image):
        image = tf.reshape(image, [1, *image.shape])
        image = tf.image.resize(image, [224, 224])
        image = tf.image.grayscale_to_rgb(image)
        image = tf.keras.applications.vgg19.preprocess_input(image)
        return image[0]

    train_imgs, train_labels, test_imgs, test_labels, id2label = readData(path, num_classes, maxEty, test_ratio)

    train_imgs = tf.data.Dataset.from_tensor_slices(train_imgs)
    test_imgs = tf.data.Dataset.from_tensor_slices(test_imgs)
    train_imgs = train_imgs.map(transform, 6)
    test_imgs = test_imgs.map(transform, 6)

    return train_imgs, train_labels, test_imgs, test_labels, id2label


def food():
    train = tfds.load('food101', split='train', shuffle_files=True)
    train = train.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32)/255, (224, 224)),
                   'label': tf.one_hot(x['label'], 101)})
    train = train.batch(32)

    test = tfds.load('food101', split='validation', shuffle_files=True)
    test = test.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32)/255, (224, 224)),
                   'label': tf.one_hot(x['label'], 101)})
    test = test.batch(32)

    return train, test


def horses_or_humans():
    train = tfds.load('horses_or_humans', split='train', shuffle_files=True)
    train = train.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32)/255, (224, 224)),
                   'label': tf.one_hot(x['label'], 2)})
    train = train.batch(32)

    test = tfds.load('horses_or_humans', split='test', shuffle_files=True)
    test = test.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32)/255, (224, 224)),
                   'label': tf.one_hot(x['label'], 2)})
    test = test.batch(32)

    return train, test


def cifar_10():
    train = tfds.load('cifar10', split='train', shuffle_files=True)
    train = train.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32) / 255, (224, 224)),
                   'label': tf.one_hot(x['label'], 10)})
    train = train.batch(32)

    test = tfds.load('cifar10', split='test', shuffle_files=True)
    test = test.map(
        lambda x: {'image': tf.image.resize(tf.cast(x["image"], tf.float32) / 255, (224, 224)),
                   'label': tf.one_hot(x['label'], 10)})
    test = test.batch(32)

    return train, test


if __name__ == '__main__':
    train, test = cifar_10()
    print(len(train))
    print(len(test))
