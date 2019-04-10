import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop

class Simplenn(object):
    def __init__(self, x_dim, n_classes, hidden_size=[32,32], batch_size=32, \
                 learning_rate=0.01, epochs=10, random_seed=None):
        model = keras.models.Sequential()
        self.x_dim = x_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_seed = random_seed
        self.model = Sequential()
        print('Start to building a neural network.')
        self.build()

    def build(self):
        self.model.add(keras.layers.Dense(units=self.hidden_size[0], \
                                     input_dim=self.x_dim,
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='relu' \
                                            ))
        self.model.add(Dropout(0.6))
        self.model.add(keras.layers.Dense(units=self.hidden_size[1], \
                                     input_dim=self.hidden_size[0],
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='tanh' \
                                            ))
        self.model.add(Dropout(0.6))
        self.model.add(keras.layers.Dense(units=self.n_classes, \
                                     input_dim=self.hidden_size[1],
                                     kernel_initializer='glorot_uniform', \
                                     bias_initializer='zeros', \
                                     activation='softmax' \
                                            ))
        sgd_optimizer = keras.optimizers.SGD(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        print('Scucceeded in building a neural network.')


    def train(self, x_train, y_train, x_valid, y_valid):
        print('Start to train.')
        self.model.fit(x_train, y_train, \
                       batch_size=self.batch_size, \
                       epochs=self.epochs, \
                       verbose=1, \
                       validation_data=(x_valid, y_valid))
        print('Scucceeded in training.')


    def test(self, x_test, y_test):
        print('Start to test.')
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Finish to test.')
