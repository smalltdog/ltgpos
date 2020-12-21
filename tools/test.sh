. ./pathcfg.sh

$NVCC --compiler-options -fPIC -DTEST" -shared \
    src/ltgpos.cu \
    src/util.cu \
    src/json_parser.cu \
    src/grid_search.cu \
    src/geodistance.cu \
    src/cJSON.c \
    -o libs/libltgpos.so

cp libs/libltgpos.so test/
g++ test/test.c -I/usr/local/cuda/include/ -L. -lltgpos -o test/test.out
./test/test.out
