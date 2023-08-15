# Neural Network Piano Piece Generator

![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

This Python project implements a neural network that generates a short piano piece. The neural network is trained on a dataset of existing piano compositions to learn the patterns and structures present in music. It then generates a new piece that reflects the learned musical style.

## Neural Network Architecture

The neural network used in this project is based on a Recurrent Neural Network (RNN) architecture, specifically a Long Short-Term Memory (LSTM) network. The LSTM is well-suited for generating sequences of data, making it suitable for creating music.

## Features

- Loading and preprocessing of musical training data.
- Neural network model construction using TensorFlow's Keras API.
- Training the model on the musical dataset.
- Generating a new piano piece using the trained model.
