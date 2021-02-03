. tools/pathcfg.sh

JAVA_BIN=$JAVA_HOME/bin

cd demo/

$JAVA_BIN/javac -encoding UTF-8 LtgposCaller.java
[ $? -eq 0 ] || exit 1 && $JAVA_BIN/javah -jni LtgposCaller
[ $? -eq 0 ] || exit 1 && $JAVA_BIN/java LtgposCaller
