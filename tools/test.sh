. tools/pathcfg.sh

$NVCC --compiler-options -fPIC -DTEST -rdc=true -shared \
    src/ltgpos.cu \
    src/grid_search.cu \
    src/json_parser.cu \
    src/geodistance.cu \
    src/configs.cu \
    src/utils.cu \
    src/cJSON.c \
    -o libs/libltgpos.so

cp libs/libltgpos.so test/
export LD_LIBRARY_PATH=test:$LD_LIBRARY_PATH
g++ test/test.c -I/usr/local/cuda/include/ -Ltest -lltgpos -o test/test.out

while getopts "i:o" arg; do
  case $arg in
    i)
      IN=$OPTARG
      ;;
    o)
      OUT=true
      ;;
  esac
done

if [ "$OUT" = true ]; then
	test/test.out $IN > test/output.csv
else
	test/test.out $IN
fi
