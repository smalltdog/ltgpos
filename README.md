# High Parallel Lightning Positioning

## Install

### 编译 .so 动态链接库，并放入库环境路径中

```shell
# 使用 nvcc 将 CUDA 代码编译成 .so 动态链接库
# 执行该 shell 脚本前需先配置 shell 脚本第 2 行和第 5 行中的路径
bash tools/make_so.sh

# 将动态链接库移动到库目录中
sudo mv libs/liblightning_position.so /usr/lib/

# # 将动态链接库移动到库目录中
# sudo mkdir /usr/lib/lightning
# sudo mv libs/liblightning_position.so /usr/lib/lightning

# # 配置 ldconfig
# sudo vim /etc/ld.so.conf.d/lightning.conf
# ############################################
# # vim 是一个文本编辑器，
# # 开启后按 i 进行编辑，此时屏幕左下角会出现 "insert"
# # 在文件中输入下行内容：
# # /etc/ld.so.conf.d/lightning.conf
# # 按 esc 退出编辑模式，输入 ":wq" 保存并退出
# ############################################

# # sudo sed -i '$a/\/usr\/lib\/lightning' /etc/ld.so.conf.d/lightning.conf
# # echo "/usr/lib/lightning" > /etc/ld.so.conf.d/lightning.conf

# # 配置动态链接器在运行时的绑定
# sudo ldconfig
```

### 编译 C 语言测试程序

```shell
g++ demo/c/test.c -I/usr/local/cuda/include/ -L. -llightning_position -o demo/c/test.out
```

### 编译并运行 Java 程序

```shell
cd demo/java

# 将 JAVA 代码编译为 JAVA 类文件
javac -encoding UTF-8 LtgPosCaller.java

# 自动生成本地方法头文件
javah -jni LtgPosCaller

# 运行 JAVA 程序
java LtgPosCaller
```

## Usage

TODO

## TODO

- [ ] 精度测试
- [ ] 写文档、代码注释
