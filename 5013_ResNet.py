import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image
import io

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置TensorFlow动态分配GPU内存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 0.定义FTML优化器
class FTML(Optimizer):
    def __init__(self, learning_rate=0.0001, beta_1=0.6, beta_2=0.999,
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

# 1. 定义CIFAR-10数据集的ImageDataGenerator类
def load_image_data(dataframe):
    # 把图片数据转换为3D Numpy数组
    dataframe['img'] = dataframe['img'].apply(lambda x: Image.open(io.BytesIO(x['bytes'])).convert('RGB'))
    dataframe['img'] = dataframe['img'].apply(lambda img: np.array(img))
    x = np.stack(dataframe['img'].values)
    y = dataframe['label'].values
    return x, y

# 2. 读取.parquet文件
def load_data(parquet_file):
    df = pd.read_parquet(parquet_file)
    return df

# 3. 定义ResNet模型
def resnet_model(input_shape, num_classes):
    base_model = keras.applications.ResNet101(weights=None, include_top=False, input_shape=input_shape)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions)
    return model

# 4. 数据预处理
def preprocess_data(x, y):
    datagen = ImageDataGenerator(
        rescale=1./255
    )
    # 设置batch size。如果显存有限，可以将batch size调低
    train_gen = datagen.flow(x, y, batch_size=128)
    return train_gen

# 5. 主函数
def main():
    # parquet_file = 'cifar10.parquet'
    # df = load_data(parquet_file)
    # x, y = load_image_data(df)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # 将标签转换为独热编码格式
    num_classes = 10
    y_one_hot = to_categorical(y_train, num_classes)
    train_gen = preprocess_data(x_train, y_one_hot)
    epochs = 200

    initial_learning_rate = 0.0001
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.0,
        staircase=True)
    
    print('rmsprop start')
    opt = optimizers.RMSprop(learning_rate=initial_learning_rate, rho=0.9, epsilon=1e-8)  
    model = resnet_model(input_shape=(32, 32, 3), num_classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, verbose=1)
    loss_values = history.history['loss']
    log_loss_values = np.log10(loss_values)
    plt.plot(log_loss_values)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    print('rmsprop end')

    print('adadelta start')
    model = resnet_model(input_shape=(32, 32, 3), num_classes=10)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, verbose=1)
    loss_values = history.history['loss']
    log_loss_values = np.log10(loss_values)
    plt.plot(log_loss_values)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    print('adadelta end')

    print('adam start')
    opt2 = Adam(learning_rate=initial_learning_rate)
    model = resnet_model(input_shape=(32, 32, 3), num_classes=10)
    model.compile(loss='categorical_crossentropy', optimizer=opt2, metrics=['accuracy'])
    history = model.fit(train_gen, epochs=epochs, verbose=1)
    loss_values = history.history['loss']
    log_loss_values = np.log10(loss_values)
    plt.plot(log_loss_values)
    K.clear_session()
    tf.compat.v1.reset_default_graph()
    print('adam end')
 
    print('ftml start')
    model = resnet_model(input_shape=(32, 32, 3), num_classes=10)
    model.compile(loss='categorical_crossentropy',
                optimizer=FTML(beta_1=0.6, beta_2=0.999, epsilon=1e-8))
    history = model.fit(train_gen, epochs=epochs, verbose=1)
    loss_values = history.history['loss']
    log_loss_values = np.log10(loss_values)
    plt.plot(log_loss_values)
    print('ftml end')

    plt.title('model loss')
    plt.ylabel('Training Loss (log10 scale)')
    plt.xlabel('Epochs')
    plt.legend(['rmsprop', 'adadelta', 'adam', 'ftml'], loc='upper right')
    plt.savefig('result_on_cifar10.jpeg')
    plt.show()


if __name__ == '__main__':
    main()