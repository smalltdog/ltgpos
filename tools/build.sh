. ./pathcfg.sh

JAVA_INC1="$JAVA_HOME/include/"
JAVA_INC2="$JAVA_HOME/include/linux"

$NVCC --compiler-options "-I$JAVA_INC1 -I$JAVA_INC2 -fPIC" -shared \
    demo/LtgPosCaller.cpp \
    src/ltgpos.cu \
    src/util.cu \
    src/json_parser.cu \
    src/grid_search.cu \
    src/geodistance.cu \
    src/cJSON.c \
    -o libs/libltgpos.so

# sudo cp libs/libltgpos.so /usr/lib/
