# ⚡️ High Parallel Lightning Positioning

high parallel lightning positioning algorithm based on nested grid search.

## Install

### 编译 .so 动态链接库，并放入库环境路径中

```shell
# 使用 nvcc 将 CUDA 代码编译成 .so 动态链接库
# 执行该 shell 脚本前需先配置 shell 脚本第 2 行和第 5 行中的路径
bash tools/so_make.sh

# # 将动态链接库移动到库目录中
# sudo mv libs/liblightning_position.so /usr/lib/
```

## Usage

### 编译并运行 C 语言测试程序

```shell
# 编译 C 代码为可执行程序
g++ demo/c/test.c -I/usr/local/cuda/include/ -L. -llightning_position -o demo/c/test.out

# 运行可执行程序
./demo/c/test.out
```

或者直接运行以下 shell 脚本

```shell
bash tools/test_make.sh
```

### 编译并运行 Java 程序

```shell
cd demo/java

# 编译 JAVA 代码为 JAVA 类文件
/path/to/your/javac -encoding UTF-8 LtgPosCaller.java

# 自动生成本地方法头文件
/path/to/your/javah -jni LtgPosCaller

# 运行 JAVA 程序
/path/to/your/java LtgPosCaller
```

或者直接运行以下 shell 脚本

```shell
bash tools/demo_make.sh
```
