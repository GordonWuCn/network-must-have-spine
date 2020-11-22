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
        
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Dropout(0.5))
        self.classifier.add(tf.keras.layers.Dense(512))
        self.classifier.add(tf.keras.layers.BatchNormalization())
        self.classifier.add(tf.keras.layers.ReLU())
        self.classifier.add(tf.keras.layers.Dropout(0.5))
        self.classifier.add(tf.keras.layers.Dense(self.num_classes))
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


    def call(self, x):
        
        # apply the conv layers
        # x = self.l1(x)
        # x = self.l2(x)
        # x = self.l3(x)
        # x = self.l4(x)

        x = self.classifier(x)

        return tf.nn.softmax(x)
        
