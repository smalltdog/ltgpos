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

## TODO
- [ ] toa map 优化版 GridSearch
- [ ] ltgpos
  - [ ] gridinv 设置方法，// TODO if area > 256, then warning.
  - [ ] gridSize和Gridinv的分配设计，尽量多，尽量算的快
  - [ ] 实现策略的时候记得修改 num_involved
  - [ ] 补函数文档
- [ ] 电流强度计算
