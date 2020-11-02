# set path to your nvcc
nvcc=/usr/local/cuda/bin/nvcc

# set path to your jdk
JAVA_HOME=/home/jhy/jdk1.8.0

JAVA_INCLUDE1="$JAVA_HOME/include/"
JAVA_INCLUDE2="$JAVA_HOME/include/linux"

# compile CUDA program
# $nvcc --compiler-options "-fPIC" -shared src/lightning_position.cu src/nested_grid_search.cu src/cJSON.c -o libs/liblightning_position.so

# compile cpp implementation of JAVA native methods with CUDA program
$nvcc --compiler-options "-I$JAVA_INCLUDE1 -I$JAVA_INCLUDE2 -DDBUG -fPIC" -shared  demo/java/LtgPosCaller.cpp src/lightning_position.cu src/nested_grid_search.cu src/cJSON.c -o libs/liblightning_position.so

sudo cp libs/liblightning_position.so /usr/lib/
