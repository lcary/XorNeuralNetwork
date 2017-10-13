#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

class TrainingData
{
public:
    TrainingData(const string filename);
    bool isEof(void)
    {
        return m_trainingDataFile.eof();
    }
    void getTopology(vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(vector<double> &inputVals);
    unsigned getTargetOutputs(vector<double> &targetOutputVals);

private:
    ifstream m_trainingDataFile;
};

void TrainingData::getTopology(vector<unsigned> &topology)
{
    string line;
    string label;

    getline(m_trainingDataFile, line);
    stringstream ss(line);
    ss >> label;
    if(this->isEof() || label.compare("topology:") != 0)
    {
        abort();
    }

    while(!ss.eof())
    {
        unsigned n;
        ss >> n;
        topology.push_back(n);
    }
    return;
}

TrainingData::TrainingData(const string filename)
{
    m_trainingDataFile.open(filename.c_str());
}


unsigned TrainingData::getNextInputs(vector<double> &inputVals)
{
    inputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss >> label;
    if (label.compare("in:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            inputVals.push_back(oneValue);
        }
    }

    return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(vector<double> &targetOutputVals)
{
    targetOutputVals.clear();

    string line;
    getline(m_trainingDataFile, line);
    stringstream ss(line);

    string label;
    ss>> label;
    if (label.compare("out:") == 0) {
        double oneValue;
        while (ss >> oneValue) {
            targetOutputVals.push_back(oneValue);
        }
    }

    return targetOutputVals.size();
}

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// ******************** class Neuron ********************

// Does the actual math for feeding forward and calculation
// of output gradients.

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;    // [0.0..1.0] overall net training weight
    static double alpha;  // [0.0..n] multiplier of last weight change (momentum)
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;   // overall net learning weight, [0.0..1.0]
double Neuron::alpha = 0.5;  // momentum, multiplier of last deltaWeight, [0.0..n]

void Neuron::updateInputWeights(Layer &prevLayer)
{
    // The weights to be updated are in the Connection container
    // in the neurons of the preceding layer (including bias neuron).

    for (unsigned n; n < prevLayer.size(); ++n) {

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


// ******************** class Net ********************

class Net
{
public:
    Net(vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backProp(const vector<double> &targetVals);
    void getResults(vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    vector<Layer> m_layers;  // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;

};

void Net::getResults(vector<double> &resultVals) const
{
    resultVals.clear();

    // Loop through all neurons in output layer and moves output
    // value onto result vals.
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n) {
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::feedForward(const vector<double> &inputVals)
{
    // Ensure number of input values is same as number of input neurons
    // (minus 1 for the bias neuron in each layer).
    assert(inputVals.size() == m_layers[0].size() - 1);

    // Assign (latch) the input values into the input neurons.
    for (unsigned i = 0; i < inputVals.size(); i++) {
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

void Net::backProp(const vector<double> &targetVals)
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
        m_error = delta * delta;
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

Net::Net(vector<unsigned> &topology)
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
            cout << "Made a layer!\n";
        }

        // Bias neuron needs to have a constant output of 1.0 value.
        // Force the last neuron created in each layer to have this output:
        m_layers.back().back().setOutputVal(1.0);
    }
}

void showVectorVals(string label, vector<double> &v)
{
    cout << label << " ";
    for(unsigned i = 0; i < v.size(); ++i)
    {
        cout << v[i] << " ";
    }
    cout << endl;
}

int main()
{
    TrainingData trainData("trainingData.txt");

    // e.g. {3, 2, 1}
    vector<unsigned> topology;
    // topology.push_back(3);
    // topology.push_back(2);
    // topology.push_back(1);

    trainData.getTopology(topology);
    Net myNet(topology);

    vector<double> inputVals, targetVals, resultVals;
    int trainingPass = 0;
    while(!trainData.isEof())
    {
        ++trainingPass;
        cout << endl << "Pass" << trainingPass;

        // Get new input data and feed it forward:
        if(trainData.getNextInputs(inputVals) != topology[0])
            break;
        showVectorVals(": Inputs :", inputVals);
        myNet.feedForward(inputVals);

        // Collect the net's actual results:
        myNet.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        // Train the net what the outputs should have been:
        trainData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        myNet.backProp(targetVals);

        // Report how well the training is working, average over recnet
        cout << "Net recent average error: "
             << myNet.getRecentAverageError() << endl;
    }

    cout << endl << "Done" << endl;
}
