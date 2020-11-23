import tensorflow as tf

class TwoConvPool(tf.keras.layers.Layer):
    def __init__(self, f1, f2):
        super(TwoConvPool, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(f1, kernel_size=3, strides=1, padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(f2, kernel_size=3, strides=1, padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()

        self.maxpooling = tf.keras.layers.MaxPooling2D(2, 2)

    def call(self, i):
        o = self.conv1(i)
        o = self.batch_norm1(o)
        o = self.relu1(o)
        
        o = self.conv2(o)
        o = self.batch_norm2(o)
        o = self.relu2(o)

        return self.maxpooling(o)

class ThreeConvPool(tf.keras.layers.Layer):
    def __init__(self, f1, f2, f3):
        super(ThreeConvPool, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(f1, kernel_size=3, strides=1, padding='same')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        
        self.conv2 = tf.keras.layers.Conv2D(f2, kernel_size=3, strides=1, padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        
        self.conv3 = tf.keras.layers.Conv2D(f3, kernel_size=3, strides=1, padding='same')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()

        self.maxpooling = tf.keras.layers.MaxPooling2D(2, 2)

    def call(self, i):
        o = self.conv1(i)
        o = self.batch_norm1(o)
        o = self.relu1(o)
        
        o = self.conv2(o)
        o = self.batch_norm2(o)
        o = self.relu2(o)
        
        o = self.conv3(o)
        o = self.batch_norm3(o)
        o = self.relu3(o)

        return self.maxpooling(o)

class LinearLayers(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(LinearLayers, self).__init__()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, i):
        o = self.dropout1(i)
        o = self.dense1(o)
        o = self.batch_norm(o)
        o = self.dropout2(o)
        o = self.dense2(o)
        return o

class SpinalLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes):
        super(SpinalLayer, self).__init__()
        self.num_classes = num_classes
        self.Half_width = 128
        self.layer_width =128
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

    def call(self, i):
        x1 = self.fc_spinal_layer1(i[:, 0:self.Half_width])
        x2 = self.fc_spinal_layer2(tf.concat([ i[:,self.Half_width:2*self.Half_width], x1], axis=1))
        x3 = self.fc_spinal_layer3(tf.concat([ i[:,0:self.Half_width], x2], axis=1))
        x4 = self.fc_spinal_layer4(tf.concat([ i[:,self.Half_width:2*self.Half_width], x3], axis=1))

        x = tf.concat([x1, x2], axis=1)
        x = tf.concat([x, x3], axis=1)
        x = tf.concat([x, x4], axis=1)
        x = self.fc_out(x)
        return tf.nn.softmax(x)

class VGG(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(0.0005)
        self.l1 = TwoConvPool(64, 64)
        self.l2 = TwoConvPool(128, 128)
        self.l3 = ThreeConvPool(256, 256, 256)
        self.l4 = ThreeConvPool(256, 256, 256)
        self.l5 = LinearLayers(num_classes)
    def call(self, i):
        o = self.l1(i)
        o = self.l2(o)
        o = self.l3(o)
        o = self.l4(o)
        o = tf.reshape(o, [o.shape[0], -1])
        o = self.l5(o)
        return o

    def accuracy_fn(self, labels, logits):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32))



class VGG_Spinal(tf.keras.Model):
    def __init__(self, num_classes):
        super(VGG_Spinal, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(0.0005)
        self.l1 = TwoConvPool(64, 64)
        self.l2 = TwoConvPool(128, 128)
        self.l3 = ThreeConvPool(256, 256, 256)
        self.l4 = ThreeConvPool(256, 256, 256)
        self.l5 = SpinalLayer(num_classes)
    def call(self, i):
        o = self.l1(i)
        o = self.l2(o)
        o = self.l3(o)
        o = self.l4(o)
        o = tf.reshape(o, [o.shape[0], -1])
        o = self.l5(o)
        return o

    def accuracy_fn(self, labels, logits):
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1)), tf.float32))