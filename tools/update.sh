. tools/pathcfg.sh

[ $? -eq 0 ] || exit 1 && rm -rf !($JAVA_HOME)
[ $? -eq 0 ] || exit 1 && cp -r ../../ $LTGPOS
[ $? -eq 0 ] || exit 1

