# ⚡️ Ltgpos

Ltgpos is a parallel lightning positioning algorithm based on grid search, with CUDA parallel computing acceleration.

## Install

### 编译 .so 动态链接库，并加入库环境路径

配置 `tools/pathcfg.sh` 中的路径，并运行以下 Shell 脚本

```shell
bash tools/build.sh
```

## Usage

### 编译并运行测试程序

运行以下 Shell 脚本

```shell
bash tools/test.sh -i /path/to/input (-o)
```

评测计算结果

```shell
python test/evaluation.py --no xxx
```

根据计算结果对输入数据进行筛选

```shell
python test/badcase.py --no xxx
```

将输出结果划分成数据与 JSON 字符串

```shell
python test/split_output.py --no xxx
```

### 编译并运行 Java Demo

运行以下 Shell 脚本

```shell
bash tools/demo.sh
```

或运行以下命令

```shell
cd demo/

# 编译 Java 代码为 Java 类文件
/path/to/javac -encoding UTF-8 LtgposCaller.java
# 自动生成本地方法头文件
/path/to/javah -jni LtgposCaller
# 运行 Java 程序
/path/to/java LtgposCaller
```

## Deploy

Included dirs: `demo/`, `src/`, `tools/`

Modify `tools/pathcfg.sh` :

```bash
# set path to your jdk
JAVA_HOME=~/ltgpos/jdk1.8.0_271

# set path to your repos
LTGPOS=~/ltgpos
```

Modify `demo/LtgposCaller.java` :

```bash
System.load("/home/yftx02/ltgpos/libs/libltgpos.so");
```

## TODO

- [ ] refer or avg?
- [ ] not minus base ms
- [ ] N & 1e6
- [ ] goodness calculation
