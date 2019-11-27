
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import keras
import utils
import statistics
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

def create_model(no_in=None, no_h1=None, no_h2=None, no_out=None):
    # Define the keras model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(no_h1, input_shape=(no_in, ), activation='tanh',  bias_initializer='ones'))
    model.add(keras.layers.Dense(no_h2, activation='tanh',  bias_initializer='ones'))
    model.add(keras.layers.Dense(no_out, activation='sigmoid'))

    # Compile the keras model
    model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    return model

def train(trainfile, testfile, model):
    # Read data from problem 2 - read_train_data_1and5(train_data)
    dataset = utils.Dataset()

    # Define train data
    train_x, train_y = dataset.read_raw_1and5(path=trainfile)
    train_y = (train_y/5).astype(int)
    train_x, train_y = shuffle(train_x, train_y)

    # Define test data
    test_x, test_y = dataset.read_raw_1and5(path=testfile)
    test_y = (test_y/5).astype(int)
    test_x, test_y = shuffle(test_x, test_y)

    in_sample_accur = []
    in_sample_loss = []
    out_sample_accur = []
    out_sample_loss = []

    # Calculate
    for i, (train_index, test_index) in enumerate(KFold(3).split(train_x)):
        print('Fold %d' % (i + 1))
        x_train, x_value = train_x[train_index], train_x[test_index]
        y_train, y_value = train_y[train_index], train_y[test_index]

        # Fit the keras model on the dataset
        model.fit(x_train, y_train, batch_size=10, epochs=20, verbose=0, shuffle=True, validation_data=(x_value, y_value))

        # Calculate in and out sample error of every fold
        scored_value = model.evaluate(x_value, y_value, verbose=0)
        in_sample_loss.append(scored_value[0])
        in_sample_accur.append(scored_value[1])
        print('In-sample error', scored_value[0], ' In-sample accuracy:', scored_value[1])

        scored_test = model.evaluate(test_x, test_y, verbose=0)
        out_sample_loss.append(scored_test[0])
        out_sample_accur.append(scored_test[1])
        print('Test set error', scored_test[0], ' Out-sample accuracy:', scored_test[1])


    # Calculate Average sample error and loss variance
    print('Final result: ')
    print('Average in-sample error:', statistics.mean(in_sample_loss), ' Average in-sample accuracy:', statistics.mean(in_sample_accur))
    print('In-sample loss variance:', statistics.variance(in_sample_loss), ' In-sample accuracy variance:', statistics.variance(in_sample_accur))

    print('Average test set error:', statistics.mean(out_sample_loss), ' Average out-sample accuracy:', statistics.mean(out_sample_accur))
    print('Test-set performance:',  statistics.variance(out_sample_loss), ' Out-sample accuracy variance:', statistics.variance(out_sample_accur))

if __name__== "__main__":
    utils.download_data()
    trainfile = 'zip.train'
    testfile = 'zip.test'

    print('\nModel [256, 6, 2, 1]: ')
    model1 = create_model(no_in=256, no_h1=6, no_h2=2, no_out=1)
    train(trainfile, testfile, model1)

    print('\nModel [256, 3, 2, 1]: ')
    model2 = create_model(no_in=256, no_h1=3, no_h2=2, no_out=1)
    train(trainfile, testfile, model2)
