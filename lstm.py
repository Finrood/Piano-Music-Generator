import glob
import pickle

import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf


def train_network():
    """ Train a Neural Network to generate music """
    print("getting notes")
    notes = get_notes()

    print("getting amount of pitch names")
    # get amount of pitch names
    n_vocab = len(set(notes))

    print("Preparing the sequences")
    network_input, network_output = prepare_sequences(notes, n_vocab)

    print("Creating the model")
    model = create_network(network_input, n_vocab)

    print("Training the model")
    train(model, network_input, network_output)


def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    notes = []

    # We load the data into an array
    for file in glob.glob("datasets/midi_songs/*mid"):
        # loading each file into a Music21 stream object using the converter.parse(file) function
        midi = converter.parse(file)
        notes_to_parse = None

        print(midi)

        parts = instrument.partitionByInstrument(midi)
        # Using that stream object we get a list of all the notes and chords in the file
        # file has instrument parts
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        # file has notes in a flat structure
        else:
            notes_to_parse = midi.flat.notes

        # We append the pitch of every note object using its string notation since
        # the most significant parts of the note can be recreated using the string notation of the pitch
        #  And we append every chord by encoding the id of every note in the chord together into a single string,
        #  with each note being separated by a dot
        # These encodings allows us to easily decode the output generated by the network into the correct notes and chords.
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('pickles/notes.pickles', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes


# Now that we have put all the notes and chords into a sequential list we can create the sequences that will serve as the input of our network.
# First, we will create a mapping function to map from string-based categorical data to integer-based numerical data.
# This is done because neural network perform much better with integer-based numerical data than string-based categorical data.
def prepare_sequences(notes, n_vocab):
    # The network will used the last 100 notes to make a prediction on which note to use next
    sequence_length = 100

    # get all pitch names
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequence and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers (lon short term memory layers)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(n_vocab)

    network_output = to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    # LSTM layers is a Recurrent Neural Net layer that takes a sequence as an input and can return either sequences (return_sequences=True) or a matrix.
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    # Dropout layers are a regularisation technique that consists of setting a fraction of input units to 0 at each update during the training to prevent overfitting. The fraction is determined by the parameter used with the layer.
    model.add(Dropout(0.3))
    # Dense layers or fully connected layers is a fully connected neural network layer where each input node is connected to each output node.
    model.add(Dense(256, activation='relu'))
    model.add(BatchNorm())
    # Parameter is fraction of input units that should be dropped during training.
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    # RMSprop optimizer as it is usually a very good choice for recurrent neural networks.
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, network_input, network_output):
    """ train the neural network """
    filepath = "models/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    # Saves best model based on loss (minimum loss) after each epoch. We only save the best model
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # For tf 2.1.0 from https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_network()