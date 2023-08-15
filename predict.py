""" This module generates notes for a midi file using the
    trained neural network """
import pickle

import numpy as np
import tensorflow as tf
from music21 import instrument, note, stream, chord
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential


# Tutorial : https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5


def generate():
    """ Generate a piano midi file """
    # load the notes used to train the model
    print("Loading pickles of midi files used to train network")
    with open('pickles/notes.pickles', 'rb') as filepath:
        notes = pickle.load(filepath)

    print("Getting all pitch names")
    # Get all pitch names
    pitchnames = sorted(set(item for item in notes))
    # Get all pitch names
    n_vocab = len(set(notes))

    print("Preparing the sequence")
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    print("Creating the network")
    model = create_network(normalized_input, n_vocab)
    print("Generating note")
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    print("Creating midi file")
    create_midi(prediction_output)


def prepare_sequences(notes, pitchnames, n_vocab):
    """ Prepare the sequences used by the Neural Network """
    # map between notes and integers and back
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    normalized_input = normalized_input / float(n_vocab)

    return network_input, normalized_input


def create_network(network_input, n_vocab):
    """ create the structure of the neural network """
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3, ))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the weights to each node
    model.load_weights('models/weights-improvement-199-0.1937-bigger.hdf5')

    return model


# We chose to generate 500 notes using the network since that is roughly two minutes of music and gives the network plenty of space to create a melody.
# For each note that we want to generate we have to submit a sequence to the network.
# The first sequence we submit is the sequence of notes at the starting index.
# For every subsequent sequence that we use as input, we will remove the first note of the sequence and insert the output of the previous iteration at the end of the sequence
# o determine the most likely prediction from the output from the network, we extract the index of the highest value.
# The value at index X in the output array correspond to the probability that X is the next note
# Then we collect all the outputs from the network into a single array.
def generate_notes(model, network_input, pitchnames, n_vocab):
    """ convert the output from the prediction to notes and create a midi file
            from the notes """
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=1)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


# First we have to determine whether the output we are decoding is a Note or a Chord.
#
# If the pattern is a Chord, we have to split the string up into an array of notes.
# Then we loop through the string representation of each note and create a Note object for each of them.
# Then we can create a Chord object containing each of these notes.
#
# If the pattern is a Note, we create a Note object using the string representation of the pitch contained in the pattern.
#
# At the end of each iteration we increase the offset by 0.5 (as we decided in a previous section) and append the Note/Chord object created to a list.
def create_midi(prediction_output):
    """ convert the output from the prediction to notes and create a midi file
            from the notes """
    # The offset between each note
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.3

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='test_output.mid')


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

    # For tf 2.1.0 from https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    generate()
