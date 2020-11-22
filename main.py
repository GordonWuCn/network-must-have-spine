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
data_path = 'quickdraw'
num_epoch = 20
batch_size = 256

loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


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

    vgg = VGG()
    vgg_fc = VGG_Spinal()

    train(vgg)

    train(vgg_fc)
