from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

class CNN(object):
    def __init__(self, vocab_size=1000, embedding_dim=256, sequence_length=256, \
                  num_filters=None, filter_size=None, drop=0.2, num_classes= 10, \
                  learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0,\
                  batch_size=32, epochs=10, verbose=1, class_weight=None):
        #assert len(num_filters) == len(filter_sizes), 'Error'
        assert len(num_filters) == 3 and len(filter_size) == 3, 'Error'
        assert num_filters and filter_size, 'Error'
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.drop = drop
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.model = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.epsilon = epsilon
        self.class_weight = class_weight
        self.decay = decay
        self.build()


    def build(self):
        print('Start to building CNN.')
        input = Input(shape=(self.sequence_length,), dtype='int32')
        embedding = Embedding(input_dim=self.vocab_size, \
                              output_dim=self.embedding_dim, \
                              input_length=self.sequence_length)(input)
        reshape = Reshape((self.sequence_length, self.embedding_dim, 1))(embedding)
        conv0 = Conv2D(self.num_filters[0], \
                       kernel_size=(self.filter_size[0], self.embedding_dim), \
                       padding='valid', \
                       kernel_initializer='normal', \
                       activation='relu')(reshape)
        conv1 = Conv2D(self.num_filters[1], \
                       kernel_size=(self.filter_size[1], self.embedding_dim), \
                       padding='valid', \
                       kernel_initializer='normal', \
                       activation='relu')(reshape)
        conv2 = Conv2D(self.num_filters[2], \
                       kernel_size=(self.filter_size[2], self.embedding_dim), \
                       padding='valid', \
                       kernel_initializer='normal', \
                       activation='relu')(reshape)
        maxpool0 = MaxPool2D(pool_size=(self.sequence_length - self.filter_size[0] + 1, 1), \
                             strides=(1,1), \
                             padding='valid')(conv0)
        maxpool1 = MaxPool2D(pool_size=(self.sequence_length - self.filter_size[1] + 1, 1), \
                            strides=(1,1), \
                            padding='valid')(conv1)
        maxpool2 = MaxPool2D(pool_size=(self.sequence_length - self.filter_size[2] + 1, 1), \
                            strides=(1,1), \
                            padding='valid')(conv2)
        concatenated_tensor = Concatenate(axis=1)([maxpool0, maxpool1, maxpool2])
        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(self.drop)(flatten)
        output = Dense(units=self.num_classes, activation='softmax')(dropout)
        self.model = Model(inputs=input, outputs=output)
        checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', \
                                      monitor='val_acc', \
                                      verbose=1, \
                                      save_best_only=True, \
                                      mode='auto')
        adam = Adam(lr=self.learning_rate, \
                    beta_1=self.beta_1, \
                    beta_2=self.beta_2, \
                    epsilon=self.epsilon, \
                    decay=self.decay)
        self.model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        print('Scucceeded in building CNN.')



    def train(self, x_train, y_train, x_valid, y_valid):
        self.model.fit(x_train, y_train, \
                       batch_size=self.batch_size, \
                       epochs=self.epochs, \
                       verbose=self.verbose, \
                       class_weight = self.class_weight, \
                       validation_data=(x_valid, y_valid))
        print('Scucceeded in training.')


    def test(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=self.verbose)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
