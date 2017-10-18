#include "neuron.hpp"

double Neuron::eta = 0.15;   // overall net learning weight, [0.0..1.0]
double Neuron::alpha = 0.5;  // momentum, multiplier of last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons of the preceding layer (including bias neuron).

    for (unsigned n = 0; n < prevLayer.size(); ++n) {

        // Convenience variable
        Neuron &neuron = prevLayer[n];

        // Other neuron's connection weight from it to us
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        // n (eta) - overall net learning rate
        //      0.0 - slow learner
        //      1.0 - medium learner
        //      2.0 - reckless learner
        // a (alpha) - momentum
        //      0.0 - no momentum
        //      0.5 - moderate momentum
        // 
        double newDeltaWeight =
                // Individual input, magnified by gradient and train rate
                eta  // common symbol used by neural net textbooks
                * neuron.getOutputVal()
                * m_gradient
                // Also add momentum = fraction of the previous delta weight
                + alpha  // common symbol used by neural net textbooks
                * oldDeltaWeight; // keeps on moving in same direction

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;

    // Sum of all contributions to errors we make to nodes we
    // feed in the next layer. Each n is a neuron in the next
    // layer, not including the bias neuron.
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
        // sum of connection weights between this neuron and
        // neuron in the next layer:
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;

}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    // To figure out error delta, look at sum of derivatives of weights
    // of next layer.
    double dow = sumDOW(nextLayer);
    // This gives us gradients if we're a hidden neuron.
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal)
{
    // Calculate difference between target value it should have,
    // and output value it actually has.
    double delta = targetVal - m_outputVal;

    // Multiply difference by derivative of output value. This
    // keeps training headed in direction where it reduces error
    // (the overall net error).
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    // We need something with a useful derivative.
    // This could be a step function or ramp function, but we need something
    // with a useful derivative for backpropagation, so we'd like a curve.

    // Use a hyperbolic tangent function
    // tanh - output range [-1.0..1.0]
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    return tanh(x);

}

double Neuron::transferFunctionDerivative(double x)
{
    // Used for back propagation learning
    // (d/dx)tanh(x) = 1 - tanh^2(x)
    return 1.0 - (x * x);
}


void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;

    // Sum the previous layer's outputs (which are our inputs).
    // Include the bias node from the previous layer.
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    // Activation (aka transfer) function
    m_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }

    m_myIndex = myIndex;
}
