import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm


def readData(path, maxEty=10000, test_ratio=.3):
    id2label = {}
    images = []
    labels = []
    for idx, file in tqdm(enumerate(os.listdir(path))):
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

    return train_inputs, train_labels, test_inputs, test_labels, id2label


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels, id2label = readData('quickdraw')
    print(train_imgs.shape)
    print(train_labels.shape)
    print(test_imgs.shape)
    print(test_labels.shape)
    print(id2label)
