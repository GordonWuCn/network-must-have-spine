import os
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SpinalVGG(tf.keras.Model):
    def __init__(self, num_classes):
        
        super(SpinalVGG, self).__init__()
        self.num_classes = num_classes
        self.Half_width = 128
        self.layer_width =128
        
        # initialize two cov pool and three cov pool
        # self.l1 = self.two_conv_pool(1, 64, 64)
        # self.l2 = self.two_conv_pool(64, 128, 128)
        # self.l3 = self.three_conv_pool(128, 256, 256, 256)
        # self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        # initialize spinal Layers
        self.fc_spinal_layer1 = tf.keras.Sequential()
        self.fc_spinal_layer1.add(tf.keras.layers.Dropout(0.5))
        self.fc_spinal_layer1.add(tf.keras.layers.Dense(self.layer_width))
        self.fc_spinal_layer1.add(tf.keras.layers.BatchNormalization())
        self.fc_spinal_layer1.add(tf.keras.layers.ReLU())

        self.fc_spinal_layer2 = tf.keras.Sequential()
        self.fc_spinal_layer2.add(tf.keras.layers.Dropout(0.5))
        self.fc_spinal_layer2.add(tf.keras.layers.Dense(self.layer_width))
        self.fc_spinal_layer2.add(tf.keras.layers.BatchNormalization())
        self.fc_spinal_layer2.add(tf.keras.layers.ReLU())

        self.fc_spinal_layer3 = tf.keras.Sequential()
        self.fc_spinal_layer3.add(tf.keras.layers.Dropout(0.5))
        self.fc_spinal_layer3.add(tf.keras.layers.Dense(self.layer_width))
        self.fc_spinal_layer3.add(tf.keras.layers.BatchNormalization())
        self.fc_spinal_layer3.add(tf.keras.layers.ReLU())

        self.fc_spinal_layer4 = tf.keras.Sequential()
        self.fc_spinal_layer4.add(tf.keras.layers.Dropout(0.5))
        self.fc_spinal_layer4.add(tf.keras.layers.Dense(self.layer_width))
        self.fc_spinal_layer4.add(tf.keras.layers.BatchNormalization())
        self.fc_spinal_layer4.add(tf.keras.layers.ReLU())

        self.fc_out = tf.keras.Sequential()
        self.fc_out.add(tf.keras.layers.Dropout(0.5))
        self.fc_out.add(tf.keras.layers.Dense(self.num_classes))
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def call(self, x):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # apply the conv layers
        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)

        x1 = self.fc_spinal_layer1(x[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(tf.concat([ x[:,self.Half_width:2*self.Half_width], x1], axis=1))
        x3 = self.fc_spinal_layer3(tf.concat([ x[:,0:self.Half_width], x2], axis=1))
        x4 = self.fc_spinal_layer4(tf.concat([ x[:,self.Half_width:2*self.Half_width], x3], axis=1))

        x = tf.concat([x1, x2], axis=1)
        x = tf.concat([x, x3], axis=1)
        x = tf.concat([x, x4], axis=1)

        x = self.fc_out(x)

        return tf.nn.softmax(x)
        
