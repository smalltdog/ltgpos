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
  - [ ] kNumSearches 改成主调函数传进去
  - [ ] gridinv 设置方法，// TODO if area > 256, then warning.
  - [ ] 计时，查看线程数，查看区域面积，格点数
  - [ ] 如有必要，再加 setcfg 函数
  - [ ] 能不能判断 gridsize 够不够大，然后进行重新setconfig，gridsize 提前判断，如果超了就自动增大，超了再恢复
  - [ ] gridSize和Gridinv的分配设计，尽量多，尽量算的快
  - [ ] 实现策略的时候记得修改 num_involved
  - [ ] 补函数文档
