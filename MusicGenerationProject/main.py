import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K


X, Y, n_values, indices_values = load_music_utils()
# print('shape of X:', X.shape= (60,30,78) )
# print('number of training examples:', X.shape[0]=60)
# print('Tx (length of sequence):', X.shape[1]=30)
# print('total # of unique values:', n_values=78)
# print('Shape of Y:', Y.shape=(30,60,78))

n_a = 64

reshapor = Reshape((1, 78))
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation='softmax')

# First generate model
def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data

    Returns:
    model -- a keras model with the
    """

    # Define the input of your model with a shape
    X = Input(shape=(Tx, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    outputs = []

    for t in range(Tx):
        # select the "t"th time step vector from X.
        x = Lambda(lambda x: X[:, t, :])(X)
        # Use reshapor to reshape x to be (1, n_values) (≈1 line); reshapor = Reshape((1, 78))
        x = reshapor(x)
        # Perform one step of the LSTM_cell; LSTM_cell = LSTM(n_a, return_state = True)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Apply densor to the hidden state output of LSTM_Cell; densor = Dense(n_values, activation='softmax')
        out = densor(a)
        # add the output to "outputs"
        outputs.append(out)

    # Create model instance
    model = Model([X, a0, c0], outputs)
    return model

# music inference
def music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """

    # Define the input of your model with a shape
    x0 = Input(shape=(1, n_values))

    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    # Loop over Ty and generate a value at every time step
    for t in range(Ty):
        # Perform one step of LSTM_cell (≈1 line); LSTM_cell = LSTM(n_a, return_state = True)
        a, _, c = LSTM_cell(x, initial_state=[a, c])

        # Apply Dense layer to the hidden state output of the LSTM_cell; densor = Dense(n_values, activation='softmax')
        out = densor(a)

        # Append the prediction "out" to "outputs". out.shape = (None, 78)
        outputs.append(out)

        # Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided
        #           the line of code you need to do this.
        x = Lambda(one_hot)(out)

    # Create model instance with the correct "inputs" and "outputs" 
    inference_model = Model([x0, a0, c0], outputs)

    return inference_model

# Music Generation
def predict_and_sample(inference_model, x_initializer=x_initializer, a_initializer=a_initializer,
                       c_initializer=c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred, 2)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=None)

    return results, indices

if __name__ == "__main__":
    # First generate model, train LSTM and densor; densor = Dense(n_values, activation='softmax'); densor is fully connected with softmax
    model = djmodel(Tx=30, n_a=64, n_values=78)
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    m = 60
    a0 = np.zeros((m, n_a))
    c0 = np.zeros((m, n_a))
    model.fit([X, a0, c0], list(Y), epochs=100)

    # Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    inference_model = music_inference_model(LSTM_cell, densor, n_values=78, n_a=64, Ty=50)

    x_initializer = np.zeros((1, 1, 78))
    a_initializer = np.zeros((1, n_a))
    c_initializer = np.zeros((1, n_a))

    # Music gereneration
    results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)

    out_stream = generate_music(inference_model)
