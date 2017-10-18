#include <cassert>
#include <fstream>
#include <iostream>
#include "neuralNet.hpp"
#include "trainingData.hpp"

using namespace std;

void showVectorVals(string label, vector<double> &v, ofstream &outfile)
{
    outfile << label << " ";
    for(unsigned i = 0; i < v.size(); ++i)
    {
        outfile << v[i] << " ";
    }
    outfile << endl;
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

    ofstream csvout;
    ofstream txtout;
    try {
        csvout.open("out/out.csv");
        txtout.open("out/out.txt");

        csvout << "TrainingPass" << "," << "RecentAverageError" << endl;

        vector<double> inputVals, targetVals, resultVals;
        int trainingPass = 0;
        while(!trainData.isEof())
        {
            ++trainingPass;
            txtout << endl << "Pass " << trainingPass;

            // Get new input data and feed it forward:
            if(trainData.getNextInputs(inputVals) != topology[0])
                break;
            showVectorVals(": Inputs :", inputVals, txtout);
            myNet.feedForward(inputVals);

            // Collect the net's actual results:
            myNet.getResults(resultVals);
            showVectorVals("Outputs:", resultVals, txtout);

            // Train the net what the outputs should have been:
            trainData.getTargetOutputs(targetVals);
            showVectorVals("Targets:", targetVals, txtout);
            assert(targetVals.size() == topology.back());

            myNet.backProp(targetVals);

            // Report how well the training is working, average over recnet
            txtout << "Net recent average error: ";
            txtout << myNet.getRecentAverageError() << endl;

            // Create csv of errors
            csvout << trainingPass << "," << myNet.getRecentAverageError() << endl;
        }
    }
    catch(...) {
        csvout.close();
        txtout.close();
    }

    cout << endl << "Done" << endl;
}
