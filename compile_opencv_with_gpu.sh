#!/bin/bash

# had been working on 2021-08-24
mkdir opencv_all
cd opencv_all
git clone https://github.com/opencv/opencv
git clone https://github.com/opencv/opencv_contrib
mkdir build
cd build
cmake -DWITH_CUDA=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
      ../opencv
cmake  --build . --parallel 12  # or whatever number of cores you want to assign for this
cd build/python_loader
sudo python3 setup.py install