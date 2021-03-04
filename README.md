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

- [ ] random shuffle in comb generation
- [ ] test for kNumNxtSchInvs
- [ ] 69 badcase analysis
- [ ] fit on 81626 data: (GoodThres, GenSchDom in GrdSch, SchDomExpRatio)
