#!/bin/bash

# Root folder
mkdir -p mini-ppti && cd mini-ppti

# Top-level files
touch README.md

# C++ structure
mkdir -p cpp/include cpp/src cpp/build
touch cpp/CMakeLists.txt
touch cpp/include/ckks_runner.h
touch cpp/include/profiler.h
touch cpp/src/ckks_runner.cpp
touch cpp/src/profiler.cpp
touch cpp/src/main.cpp

# Python structure
mkdir -p python
touch python/graph.py
touch python/ops.py
touch python/partition.py
touch python/fusion.py
touch python/cost_model.py
touch python/run_experiment.py

# Configs
mkdir -p configs
touch configs/toy_mlp.json
touch configs/toy_transformer_block.json

# Results
mkdir -p results/logs
mkdir -p results/plots

# Notes
mkdir -p notes
touch notes/design.md
touch notes/experiments.md

echo "✅ mini-ppti project structure created successfully!"