#ifndef NEURALNET_H
#define NEURALNET_H

#include "neuron.hpp"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <vector>

// ******************** class Net ********************

class Net
{
public:
    Net(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }

private:
    std::vector<Layer> m_layers;  // m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    static double m_recentAverageSmoothingFactor;

};

#endif /* NEURALNET_H */
