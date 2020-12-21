# ⚡️ Ltgpos

Ltgpos is a high parallel lightning positioning algorithm based on nested grid search, with CUDA parallel computing acceleration.

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
bash tools/test.sh
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
/path/to/your/javac -encoding UTF-8 LtgposCaller.java
# 自动生成本地方法头文件
/path/to/your/javah -jni LtgposCaller
# 运行 Java 程序
/path/to/your/java LtgposCaller
```
