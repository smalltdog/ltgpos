# High Parallel Lightning Positioning

## Install

### 编译 CUDA 代码成 .so 动态链接库，并放入库环境路径中

```shell
# 使用 nvcc 将 CUDA 代码编译成 .so 动态链接库
bash tools/make.sh

# 将动态链接库移动到库目录中
sudo mkdir /usr/lib/lightning
sudo mv libs/liblightning_position.so /usr/lib/lightning

# 配置 ldconfig
sudo vim /etc/ld.so.conf.d/lightning.conf
############################################
# vim 是一个文本编辑器，
# 开启后按 i 进行编辑，此时屏幕左下角会出现 "insert"
# 在文件中输入下行内容：
# /etc/ld.so.conf.d/lightning.conf
# 按 esc 退出编辑模式，输入 ":wq" 保存并退出
############################################

# sudo sed -i '$a/\/usr\/lib\/lightning' /etc/ld.so.conf.d/lightning.conf
# echo "/usr/lib/lightning" > /etc/ld.so.conf.d/lightning.conf

# 配置动态链接器在运行时的绑定
sudo ldconfig
```

### 编译 C 语言测试程序

```shell
g++ test.c -I/usr/local/cuda/include/ -L. -llightning_position -o test.out
```

### 编译 Java 程序

```shell
cd build

# 将 .java 编译为 .class 文件
javac -encoding UTF-8 LtgPosCaller.java

# 自动生成本地方法头文件 .h
javah -jni LtgPosCaller

# 
gcc -I/usr/lib/jvm/java-7-openjdk-amd64/include -I/usr/lib/jvm/java-7-openjdk-amd64/include/solaris -fPIC -shared -o libHelloWorld.so HelloWorld.c
```

## Usage

TODO

## TODO

- [ ] 精度测试
- [ ] 写文档、代码注释