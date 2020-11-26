from comet_ml import Experiment

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from model import VGG, VGG_Spinal, VGG_Transfer_Spinal
from preprocess import scratch_data, transfer_data, food, horses_or_humans

from tqdm import tqdm

experiment = Experiment(log_code=True)

'''
hyper params
'''
# data_path = '/home/gordonwu/Downloads/quickdraw'
data_path = 'quickdraw'
num_epoch = 20
batch_size = 10
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
                if idx % 1000 == 0:
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


def train_food():
    model = VGG_Transfer_Spinal(101)
    train_imgs, test_imgs = food()
    for epoch in range(num_epoch):
        train_imgs.shuffle(1024)
        with experiment.train():
            for example in tqdm(train_imgs):
                with tf.GradientTape() as tape:
                    logits = model(example["image"])
                    loss = loss_fn(example['label'], logits)
                gradient = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                experiment.log_metric('loss', loss)
        if epoch % 5 == 4:
            test_food(model, test_imgs)


def test_food(model, test_imgs):
    with experiment.test():
        accuracy_sum = 0
        batch_cnt = 0
        for example in tqdm(test_imgs):
            logits = model(example["image"])
            accuracy_sum += model.accuracy_fn(example['label'], logits)
            batch_cnt += 1
        accuracy = accuracy_sum / batch_cnt
        experiment.log_metric('accuracy', accuracy)
    print(accuracy)


def train_hh():
    model = VGG_Transfer_Spinal(2)
    train_imgs, test_imgs = horses_or_humans()
    for epoch in range(num_epoch):
        train_imgs.shuffle(1024)
        with experiment.train():
            for example in tqdm(train_imgs):
                with tf.GradientTape() as tape:
                    logits = model(example["image"])
                    loss = loss_fn(example['label'], logits)
                gradient = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
                experiment.log_metric('loss', loss)
        if epoch % 5 == 4:
            test_hh(model, test_imgs)


def test_hh(model, test_imgs):
    with experiment.test():
        accuracy_sum = 0
        batch_cnt = 0
        for example in tqdm(test_imgs):
            logits = model(example["image"])
            accuracy_sum += model.accuracy_fn(example['label'], logits)
            batch_cnt += 1
        accuracy = accuracy_sum / batch_cnt
        experiment.log_metric('accuracy', accuracy)
    print(accuracy)


if __name__ == '__main__':
    # train_imgs, train_labels, test_imgs, test_labels, _ = scratch_data(data_path, num_of_classes, 100)
    # train_imgs, train_labels, test_imgs, test_labels, _ = transfer_data(data_path, num_of_classes, 100)

    # image = train_imgs[0][:,:,0]
    # print(image.numpy())
    # plt.imshow(image.numpy() * 255, cmap='gray')
    # plt.show()
    # mnist = tf.keras.datasets.mnist
    # (train_imgs, train_labels),(test_imgs, test_labels) = mnist.load_data()
    # train_imgs = train_imgs / 256
    # test_imgs = test_imgs / 256
    # train_labels = tf.one_hot(train_labels, 10)
    # test_labels = tf.one_hot(test_labels, 10)
    # train_imgs = tf.reshape(train_imgs, [*(train_imgs.shape), 1])
    # test_imgs = tf.reshape(test_imgs, [*(test_imgs.shape), 1])

    # vgg = VGG(num_of_classes)
    # vgg_spinal = VGG_Spinal(num_of_classes)
    # vgg_transfer_spinal = VGG_Transfer_Spinal(num_of_classes)
    #
    # train(vgg_transfer_spinal)
    # train(vgg_fc)
    # train_food()
    train_hh()
