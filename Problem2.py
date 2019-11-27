from sklearn.model_selection import KFold
from sklearn.utils import shuffle
import keras
import numpy
import utils
import statistics

def read_train_data_1and5(train_data):
    train_data_1and5 = numpy.empty([len(train_data[1] + train_data[5]), 3], dtype=float)
    for i, d in enumerate(train_data[1]):
        train_data_1and5[i][0] = d['name']
        train_data_1and5[i][1] = d['decimal']
        train_data_1and5[i][2] = d['symm']
    for i, d in enumerate(train_data[5]):
        train_data_1and5[i+len(train_data[1])][0] = d['name']
        train_data_1and5[i+len(train_data[1])][1] = d['decimal']
        train_data_1and5[i+len(train_data[1])][2] = d['symm']
    return train_data_1and5

def create_model(no_in=None, no_out=None, no_hidden=None):
    # Define keras model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(no_hidden, input_shape=(no_in, ), activation='tanh',  bias_initializer='ones'))
    model.add(keras.layers.Dense(no_out, activation='softmax'))
    
    # Compile keras model
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
    return model

def train(trainfile, testfile, model):
    # Read data
    dataset = utils.Dataset()
    train_data = dataset.read_data(path=trainfile)
    test_data = dataset.read_data(path=testfile)

    # Define train data
    train_data = read_train_data_1and5(train_data)
    train_y = train_data[:,0]
    train_y = (train_y/5).astype(int)
    train_x = train_data[:,1:3].astype('float32')
    train_x, train_y = shuffle(train_x, train_y)

    # Define test data
    test_data = read_train_data_1and5(test_data)
    test_y = test_data[:,0]
    test_y = (test_y/5).astype(int)
    test_x = test_data[:,1:3].astype('float32')
    test_x, test_y = shuffle(test_x, test_y)

    # Convert class vectors to binary class matrices
    train_y = keras.utils.to_categorical(train_y, 2)
    test_y = keras.utils.to_categorical(test_y, 2)

    in_sample_accur = []
    in_sample_loss = []
    out_sample_accur = []
    out_sample_loss = []

    # Calculate for each hidden unit 
    for i, (train_index, test_index) in enumerate(KFold(3).split(train_x)):
        print('Fold %d' % (i + 1))
        x_train, x_value = train_x[train_index], train_x[test_index]
        y_train, y_value = train_y[train_index], train_y[test_index]

        # Fit the keras model to the dataset
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

    # Calculate Average sample error
    print('Final result: ')
    print('Average in-sample error:', statistics.mean(in_sample_loss), ' Average in-sample accuracy:', statistics.mean(in_sample_accur))
    # print('In-sample loss variance:', statistics.variance(in_sample_loss), ' In-sample accuracy variance:', statistics.variance(in_sample_accur))

    print('Average test set error:', statistics.mean(out_sample_loss), ' Average out-sample accuracy:', statistics.mean(out_sample_accur))
    # print('Test-set performance:',  statistics.variance(out_sample_loss), ' Out-sample accuracy variance:', statistics.variance(out_sample_accur))


if __name__== "__main__":
    utils.download_data()
    trainfile = 'features.train'
    testfile = 'features.test'

    # Describe and plot network structure
    print('\nNumber of hidden unit = 1:')
    model1 = create_model(no_in=2, no_out=2, no_hidden=1)
    train(trainfile, testfile, model1)

    print('\nNumber of hidden unit = 5:')
    model2 = create_model(no_in=2, no_out=2, no_hidden=5)
    train(trainfile, testfile, model2)

    print('\nNumber of hidden unit = 10:')
    model3 = create_model(no_in=2, no_out=2, no_hidden=10)
    train(trainfile, testfile, model3)
