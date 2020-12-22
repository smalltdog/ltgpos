. tools/pathcfg.sh

JAVA_INC1="$JAVA_HOME/include/"
JAVA_INC2="$JAVA_HOME/include/linux"

$NVCC --compiler-options "-I$JAVA_INC1 -I$JAVA_INC2 -fPIC" -rdc=true -shared \
    demo/LtgposCaller.cpp \
    src/ltgpos.cu \
    src/json_parser.cu \
    src/grid_search.cu \
    src/geodistance.cu \
    src/utils.cu \
    src/cJSON.c \
    -o libs/libltgpos.so

# sudo cp libs/libltgpos.so /usr/lib/
