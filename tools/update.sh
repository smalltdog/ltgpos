. tools/pathcfg.sh

CUR=`pwd`
[ $? -eq 0 ] || exit 1 && cd $LTGPOS
[ $? -eq 0 ] || exit 1 && rm -rf demo src tools
cd $CUR
[ $? -eq 0 ] || exit 1 && cp -r ./* $LTGPOS
[ $? -eq 0 ] || exit 1

cd $LTGPOS
sh tools/build.sh
[ $? -eq 0 ] || exit 1 && echo "==> Updated to Ltgpos v2.1.8."
