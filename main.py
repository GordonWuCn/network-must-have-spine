from comet_ml import Experiment

import tensorflow as tf
import numpy as np

from model import VGG, VGG_Spinal
from preprocess import readData

from tqdm import tqdm

experiment = Experiment(log_code=True)

'''
hyper params
'''
data_path = '/home/gordonwu/Downloads/quickdraw'
num_epoch = 20
batch_size = 256
num_of_classes = 24

loss_fn = tf.keras.losses.CategoricalCrossentropy()



def train(model):
    for epoch in range(num_epoch):
        with experiment.train():
            for idx in tqdm(range(0, train_imgs.shape[0] - batch_size, batch_size)):
                data_batch = train_imgs[idx: idx+batch_size]
                label_batch = train_labels[idx: idx+batch_size]

                with tf.GradientTape() as tape:
                    logits = model(data_batch)
                    
                    loss = loss_fn(label_batch, logits)
                gradient = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

                experiment.log_metric('loss', loss)
                if idx % 10 == 0:
                    print(loss)
        if epoch % 5 == 4:
            print(test(model))


def test(model):
    with experiment.test():
        accuracy_sum = 0
        batch_cnt = 0
        for idx in tqdm(range(0, test_imgs.shape[0] - batch_size, batch_size)):
            data_batch = test_imgs[idx: idx + batch_size]
            label_batch = test_labels[idx: idx + batch_size]
            logits = model(data_batch)

            accuracy_sum += model.accuracy_fn(label_batch, logits)
            batch_cnt += 1

        accuracy = accuracy_sum / batch_cnt
        experiment.log_metric('accuracy', accuracy)
    return accuracy


if __name__ == '__main__':
    train_imgs, train_labels, test_imgs, test_labels, _ = readData(data_path)
    print(train_imgs.shape)
    print(train_labels.shape)
    train_imgs = tf.reshape(train_imgs, [train_imgs.shape[0], 28, 28, 1])
    train_imgs = train_imgs / 256
    test_imgs = test_imgs / 256
    train_labels = tf.one_hot(train_labels, num_of_classes)
    test_labels = tf.one_hot(test_labels, num_of_classes)
    # print(train_labels[0:100,:])    

    # mnist = tf.keras.datasets.mnist
    # (train_imgs, train_labels),(test_imgs, test_labels) = mnist.load_data()
    # train_imgs = train_imgs / 256
    # test_imgs = test_imgs / 256
    # train_labels = tf.one_hot(train_labels, 10)
    # test_labels = tf.one_hot(test_labels, 10)
    # train_imgs = tf.reshape(train_imgs, [*(train_imgs.shape), 1])
    # test_imgs = tf.reshape(test_imgs, [*(test_imgs.shape), 1])
    vgg = VGG(num_of_classes)
    # vgg_fc = VGG_Spinal()

    train(vgg)
    # test(vgg)
    # train(vgg_fc)
