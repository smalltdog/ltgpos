bash tools/so_make.sh
cp libs/liblightning_position.so demo/c/

g++ demo/c/test.c -I/usr/local/cuda/include/ -L. -llightning_position -o demo/c/test.out

./demo/c/test.out
