#!/bin/bash

set -e

# cd to repo root
cd $(git rev-parse --show-toplevel)

mkdir -p out/
g++ ./makeTrainingSamples.cpp -o out/makeTrainingSamples
out/makeTrainingSamples > trainingData.txt
g++ neuralNet.cpp -o out/neuralNet
out/neuralNet

python graphit.py out/out.csv out/out.png
open out/out.png
