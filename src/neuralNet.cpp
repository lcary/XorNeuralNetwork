#include "neuralNet.hpp"

double Net::m_recentAverageSmoothingFactor = 100.0;

void Net::getResults(std::vector<double> &resultVals) const
{
    resultVals.clear();

    // Loop through all neurons in output layer and moves output
    // value onto result vals.
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const std::vector<double> &targetVals)
{
    // Calculate overall net error (RMS of output neuron errors).
    // RMS = "Root Mean Squared Errors".

    // Handle to output layer for readability:
    Layer &outputLayer = m_layers.back();

    // Accumulate overall net error (sum of squares of errors) in a variable:
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        // target minus actual is error delta
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
    }

    m_error /= outputLayer.size() - 1;  // get average error squared
    m_error = sqrt(m_error);  // RMS

    // This has nothing to do with neural net itself, but helps
    // print out an error indication of how well that the net has
    // been doing over the last several dozen training samples
    // (how well the net is being trained).
    m_recentAverageError = 
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    // Calculate output layer gradients.
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // Calculate gradients on hidden layers.
    // (Don't include input layer, so layers > 0).
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
        // convenience variables for documentation purposes
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers[layerNum + 1];

        // Loop through hidden layer neurons and calculate gradients
        for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }

    }

    // For all layers from outputs to first hidden layer,
    // update connection weights.
    // (Don't include input layer, so layers > 0).
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
        // convenience variables for documentation purposes
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        // Update input weights from previous layer for all neurons in layer
        for (unsigned n = 0; n < layer.size() - 1; ++n) {
            layer[n].updateInputWeights(prevLayer);
        }
    }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
    // Ensure number of input values is same as number of input neurons
    // (minus 1 for the bias neuron in each layer).
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons.
    for (unsigned i = 0; i < inputVals.size(); ++i) {
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    // Forward propagation (start with first layer, skipping inputs).
    // Iterate through neurons, skipping biased neuron.
    // Address an individual neuron to update output value (maths).
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) {
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const std::vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {

        // Create new layer
        m_layers.push_back(Layer());

        // Outputs for each neuron is 0 for last layer,
        // Otherwise, it's the number of neurons in the next layer.
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Fill layer with neurons (add one extra biased neuron to each layer)
        for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));
        }

        // Bias neuron needs to have a constant output of 1.0 value.
        // Force the last neuron created in each layer to have this output:
        m_layers.back().back().setOutputVal(1.0);
    }
}
