. tools/pathcfg.sh

[ $? -eq 0 ] || exit 1
if [ "`ls -A libs`" != "" ]; then
    rm libs/*
fi
[ $? -eq 0 ] || exit 1


JAVA_INC1="$JAVA_HOME/include/"
JAVA_INC2="$JAVA_HOME/include/linux"

$NVCC --compiler-options "-I$JAVA_INC1 -I$JAVA_INC2 -fPIC" -rdc=true -shared \
    demo/LtgposCaller.cpp \
    src/ltgpos.cu \
    src/grid_search.cu \
    src/json_parser.cu \
    src/geodistance.cu \
    src/configs.cu \
    src/utils.cu \
    src/cJSON.c \
    -o libs/libltgpos.so

# sudo cp libs/libltgpos.so /usr/lib/
