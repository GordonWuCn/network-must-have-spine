from comet_ml import Experiment

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from model import VGG, VGG_Spinal, VGG_Transfer_Spinal, DenseNet_Transfer_Spinal
from preprocess import scratch_data, transfer_data, food, horses_or_humans, cifar_10, transfer_data_large, scratch_data_large

from tqdm import tqdm

experiment = Experiment(log_code=True)

'''
hyper params
'''
# data_path = '/home/gordonwu/Downloads/quickdraw'
data_path = '/home/gordonwu0722/quickdraw'
num_epoch = 100
batch_size = 100
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
        if epoch == 19:
            model.pretrained.trainable = True
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

def train_large(model):
    for epoch in range(num_epoch):
        with experiment.train():
            for idx, data_batch in tqdm(enumerate(train_imgs), total = 1680):
                label_batch = train_labels[idx * batch_size: idx * batch_size + batch_size]
                with tf.GradientTape() as tape:
                    logits = model(data_batch)
                    loss = loss_fn(label_batch, logits)
                gradient = tape.gradient(loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

                experiment.log_metric('loss', loss)
                if idx % 10 == 0:
                    print(loss)
        model.save_weights('./checkpoints')
        if epoch % 5 == 4:
            print(test_large(model))
        if epoch == 19:
            model.pretrained.trainable = True
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

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

def test_large(model):
    with experiment.test():
        accuracy_sum = 0
        batch_cnt = 0
        for idx, data_batch in tqdm(enumerate(test_imgs)):
            label_batch = test_labels[idx * batch_size: idx * batch_size + batch_size]
            logits = model(data_batch)

            accuracy_sum += model.accuracy_fn(label_batch, logits)
            batch_cnt += 1

        accuracy = accuracy_sum / batch_cnt
        experiment.log_metric('accuracy', accuracy)
    return accuracy


def train_transfer(model, train_imgs, test_imgs):
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
            test_transfer(model, test_imgs)
        if epoch == 19:
            model.pretrained.trainable = True
            model.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)


def test_transfer(model, test_imgs):
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


def train_food(is_spinal=True):
    model = VGG_Transfer_Spinal(101, is_spinal, 25088//2)
    train_imgs, test_imgs = food()
    train_transfer(model, train_imgs, test_imgs)


def train_hh(is_spinal=True):
    model = VGG_Transfer_Spinal(2, is_spinal, 25088//2)
    train_imgs, test_imgs = horses_or_humans()
    train_transfer(model, train_imgs, test_imgs)


def train_cifar_10(is_spinal=True):
    model = DenseNet_Transfer_Spinal(10, is_spinal, 512, 20)
    train_imgs, test_imgs = cifar_10()
    train_transfer(model, train_imgs, test_imgs)


if __name__ == '__main__':
    # train_imgs, train_labels, test_imgs, test_labels, _ = scratch_data(data_path, num_of_classes, 10000)
    # train_imgs, train_labels, test_imgs, test_labels, _ = scratch_data_large(data_path, num_of_classes, 10000)
    train_imgs, train_labels, test_imgs, test_labels, _ = transfer_data_large(data_path, num_of_classes, 10000)
    train_imgs = train_imgs.batch(batch_size)
    test_imgs = test_imgs.batch(batch_size)
    train_imgs = train_imgs.prefetch(5)
    # for j, datas in enumerate(train_imgs):
    #     for i, data in enumerate(datas):
    #         image = data[:,:,0]
    #         label_batch = train_labels[j: j+batch_size]
    #         print(label_batch[i])
    #         plt.imshow(image.numpy() * 256, cmap='gray')
    #         plt.show()
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
    vgg_transfer_spinal = VGG_Transfer_Spinal(num_of_classes, True, 25088//2, 1024)
    # train_large(vgg_spinal)
    # train(vgg_spinal)
    # vgg_transfer_spinal.load_weights('./checkpoints')
    train_large(vgg_transfer_spinal)
    #print(test_large(vgg_transfer_spinal))
    vgg_transfer_spinal.save_weights('./checkpoints')
    #test_large(vgg_transfer_spinal)
    # train(vgg_fc)
    # train_food()
    # train_hh()
    # train_cifar_10()
