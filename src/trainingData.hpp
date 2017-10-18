#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// ******************** class TrainingData ********************

// Reads training data and gets topology to train neural net.

class TrainingData
{
public:
    TrainingData(const std::string filename);
    bool isEof(void)
    {
        return m_trainingDataFile.eof();
    }
    void getTopology(std::vector<unsigned> &topology);

    // Returns the number of input values read from the file:
    unsigned getNextInputs(std::vector<double> &inputVals);
    unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
    std::ifstream m_trainingDataFile;
};

#endif /* TRAINING_DATA_H */
