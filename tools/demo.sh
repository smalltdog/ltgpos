. ./pathcfg.sh

JAVA_BIN=$JAVA_HOME/bin

cd demo/

$JAVA_BIN/javac -encoding UTF-8 LtgposCaller.java
$JAVA_BIN/javah -jni LtgposCaller
$JAVA_BIN/java LtgposCaller
