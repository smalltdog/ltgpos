nvcc=/usr/local/cuda-10.0/bin/nvcc
nvcc --compiler-options "-fPIC" -shared lightning_position.cu nested_grid_search.cu cJSON.c -o liblightning_position.so
g++ test.c -I/usr/local/cuda/include/ -L. -llightning_position -o test.out