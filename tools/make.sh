nvcc=/usr/local/cuda-10.0/bin/nvcc

JAVA_HOME=/home/jhy/jdk1.8.0_271 
JAVA_INCLUDE1="$JAVA_HOME/include/"
JAVA_INCLUDE2="$JAVA_HOME/include/linux"

# compile cuda program
# $nvcc --compiler-options "-fPIC" -shared src/lightning_position.cu src/nested_grid_search.cu src/cJSON.c -o libs/liblightning_position.so

# compile cpp implementation of JAVA native methods
$nvcc --compiler-options "-I$JAVA_INCLUDE1 -I$JAVA_INCLUDE2 -DDBUG -fPIC" -shared build/LtgPosCaller.cpp src/lightning_position.cu src/nested_grid_search.cu src/cJSON.c -o libs/liblightning_position.so
