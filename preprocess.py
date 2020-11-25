import numpy as np
import tensorflow as tf
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
    test_imgs = tf.concat([test_imgs] * 3, axis=3)
    test_imgs = tf.keras.applications.vgg19.preprocess_input(test_imgs)

    return train_imgs, train_labels, test_imgs, test_labels, id2label


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels, id2label = transfer_data('quickdraw', 24, 1)
    for idx in range(train_labels.shape[1]):
        if train_labels[0, idx] == 1:
            print(id2label[idx])
    plt.imshow(train_imgs[0])
    plt.show()
