Simple Neural Net Project
=========================

Overview
--------

This is a simple artificial neural network designed to improve performance on training an XOR function from scratch. This is inspired by the [“Neural Net in C++ Tutorial” by David Miller](https://vimeo.com/19569529).

Assumptions: the neural network is fully connected, implicitly connected, forward connected. Each neuron is fully connected to each neuron to the “right” (closer to the output) of it.

Features: back propagation as the method, gradient descent as the algorithm, and adjustable momentum.

XOR Function
------------

XOR = Exclusive OR

(This is a "Hello World" function for testing our neural net)

Expected input to output values:

|In 0|In 1|Out|
|----|----|---|
|0   |0   |0  |
|0   |1   |1  |
|1   |0   |1  |
|1   |1   |0  |
