
import tensorflow as tf
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Activation, ZeroPadding2D,Input,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
one_hot = np.eye(10)[y_train]
# one_hot = np.eye(10)[[y[0] for y in y_train]]


class FTML(Optimizer):
    def __init__(self, learning_rate=0.002, beta_1=0.6, beta_2=0.999,
                 epsilon=1e-8, decay=0., name='FTML', **kwargs):
        super(FTML, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("decay", decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'z')
            self.add_slot(var, 'v')
            self.add_slot(var, 'd')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        t = self.iterations + 1
        t_float32 = tf.cast(t, tf.float32)
        lr_t = lr_t / (1. - tf.pow(self.beta_1,t_float32 ))
        z = self.get_slot(var, 'z')
        v = self.get_slot(var, 'v')
        d = self.get_slot(var, 'd')

        beta_1_t = self._get_hyper('beta_1', var_dtype)
        beta_2_t = self._get_hyper('beta_2', var_dtype)

        v_t = beta_2_t * v + (1. - beta_2_t) * tf.square(grad)
        d_t = (tf.sqrt(v_t / (1. - tf.pow(beta_2_t, t_float32))) + self.epsilon) / lr_t
      
        sigma_t = d_t - beta_1_t * d
        z_t = beta_1_t * z + (1. - beta_1_t) * grad - sigma_t * var
        p_t = - z_t / d_t
        
        var_update = var.assign(p_t)
      
        updates = [z.assign(z_t), v.assign(v_t), d.assign(d_t), var_update]

        return tf.group(*updates)

    def get_config(self):
        config = super(FTML, self).get_config()
        config.update({
            'learning_rate': self._serialize_hyperparameter("learning_rate"),
            'decay': self._serialize_hyperparameter("decay"),
            'beta_1': self._serialize_hyperparameter("beta_1"),
            'beta_2': self._serialize_hyperparameter("beta_2"),
            'epsilon': self.epsilon
        })
        return config


img_rows, img_cols = 28, 28
# img_rows, img_cols = 32,32
input_shape = (img_rows, img_cols, 1)
# input_shape = (img_rows, img_cols, 3)

X = x_train
y = one_hot


def train_model(shape=(28, 28, 1), num_classes=10):
    X_input = Input(shape=shape)
    X = Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               kernel_initializer='he_normal',
               input_shape=input_shape)(X_input)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64, (3, 3), activation='relu')(X)
    X = MaxPool2D(pool_size=(2, 2))(X)
    X = Dropout(0.3)(X)
    X = Conv2D(128, (3, 3), activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Flatten()(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(num_classes, activation='softmax')(X)

    model = Model(inputs=X_input, outputs=X, name="CNN")
    return model

epochs = 30

batch_size = 256
initial_learning_rate = 0.001
lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0,
    staircase=True)
opt = optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, epsilon=1e-8)


# print('rmsprop start')
# model = train_model()
# # model = train_model(shape=(32,32,3))
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# history = model.fit(X,y,batch_size=batch_size, epochs=epochs, verbose=1)
# plt.plot(history.history['loss'])
# print('rmsprop end')

# print('adadelta start')
# model = train_model()
# # model = train_model(shape=(32,32,3))
# opt3 = tf.keras.optimizers.Adadelta(learning_rate=0.8, rho=0.8)
# model.compile(loss='categorical_crossentropy', optimizer=opt3, metrics=['accuracy'])
# history = model.fit(X,y,batch_size=batch_size, epochs=epochs, verbose=1)
# plt.plot(history.history['loss'])
# print('adadelta end')

print('adam start')
model = train_model()
# model = train_model(shape=(32,32,3))
opt2 = Adam(learning_rate=initial_learning_rate)  
model.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])
history = model.fit(X,y,batch_size=batch_size, epochs=epochs, verbose=1)
plt.plot(history.history['loss'])
print('adam end')

print('ftml start')
x_train, y_train = X, y
model = train_model()
# model = train_model(shape=(32,32,3))
model.compile(loss='categorical_crossentropy',
              optimizer=FTML(beta_1=0.6, beta_2=0.999, epsilon=1e-8))
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
plt.plot(history.history['loss'])
print('ftml end')

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['rmsprop', 'adadelta', 'adam', 'ftml'], loc='upper right')
# plt.ylim(0, 0.1)
# plt.ylim(0.6, 1.2)
# plt.xlim(15, 30)
plt.savefig('plot2.jpeg')
# plt.savefig('plot3.jpeg')
plt.show()
