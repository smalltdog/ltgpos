$nvcc=/usr/local/cuda-10.0/bin/nvcc
$nvcc --compiler-options "-fPIC" -shared ../src/lightning_position.cu ../src/nested_grid_search.cu ../src/cJSON.c -o ../libsliblightning_position.so