JAVA_HOME=~/jdk1.8.0/bin

cd demo/java

$JAVA_HOME/javac -encoding UTF-8 LtgPosCaller.java
$JAVA_HOME/javah -jni LtgPosCaller
$JAVA_HOME/java LtgPosCaller