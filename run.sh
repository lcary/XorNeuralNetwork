#!/bin/bash

set -e

# cd to repo root
cd $(git rev-parse --show-toplevel)

mkdir -p out/
g++ src/makeTrainingSamples.cpp -o out/makeTrainingSamples
out/makeTrainingSamples > trainingData.txt
g++ src/main.cpp src/neuron.cpp src/neuralNet.cpp src/trainingData.cpp -o out/neuralNet
out/neuralNet

python scripts/graphit.py out/out.csv out/out.png
open out/out.png
