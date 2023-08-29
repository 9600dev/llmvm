#!/bin/bash
git submodule update --init --recursive
poetry install

# make faiss for cuda
cd faiss 
rm -rf build/
rm -rf CMakeCache.txt
cmake -DFAISS_ENABLE_GPU=ON -DCUDAToolkit_INCLUDE_DIR=/usr/include -DCUDAToolkit_ROOT=/usr/lib/cuda -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -B build .

make -c build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python setup.py install

# make llama-cpp-python for cuda 
cd ../../../llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDAToolkit_INCLUDE_DIR=/usr/include -DCUDAToolkit_ROOT=/usr/lib/cuda" FORCE_CMAKE=1 VERBOSE=1 pip install .[server] -v --force-reinstall

