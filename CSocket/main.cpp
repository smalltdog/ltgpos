#include <iostream>
#include <stdio.h>
#include <vector>
#include <winsock2.h>
#include <fstream>
#include <direct.h>
#include <ctime>
using namespace std;
#pragma comment(lib, "ws2_32.lib")

vector<string> split(const string& str, const string& delim) {
    vector<string> res;
    if("" == str) return res;
    //先将要切割的字符串从string类型转换为char*类型
    char * strs = new char[str.length() + 1] ; //不要忘了
    strcpy(strs, str.c_str());

    char * d = new char[delim.length() + 1];
    strcpy(d, delim.c_str());

    char *p = strtok(strs, d);
    while(p) {
        string s = p; //分割得到的字符串转换为string类型
        res.push_back(s); //存入结果数组
        p = strtok(NULL, d);
    }

    return res;
}

int main() {
    // 初始化WSA
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2,2), &wsaData);
    // 创建套接字
    SOCKET slisten = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(slisten == INVALID_SOCKET){
        printf("Init error\n");
        return 0;
    }

    // 绑定IP和端口

    sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port = htons(8888);  // 绑定端口
//    sin.sin_addr.S_un.S_addr = INADDR_ANY;
    sin.sin_addr.S_un.S_addr = inet_addr("127.0.0.1");;
    if(bind(slisten, (LPSOCKADDR)&sin, sizeof(sin)) == SOCKET_ERROR)
    {
//        LPSOCKADDR 就是 sockaddr*
        printf("bind error !");
        closesocket(slisten);
        return 0;
    }
    else
        listen(slisten, 5);
//    开始监听
    SOCKET sClient;

    const char * sendData = "DataReceived!\n";
    int i = 0, n = 10;
//    超过n次连接失败终止程序
    while(true){
        sockaddr_in remoteAddr;
        int nAddrLen = sizeof(remoteAddr);
        char revData[4096];
        printf("\n Waiting for connection......\n");
        sClient = accept(slisten, (SOCKADDR *)&remoteAddr, &nAddrLen);
        if(sClient == INVALID_SOCKET){
            printf("Accept error!\n");
            if(i++ > n){
                break;
            }
            continue;
        }
        printf("One connection detected!\n");
        // 创建供写入的文件
        time_t rawTime;
        struct tm *info;
        time(&rawTime);
        info = localtime(&rawTime);
        char* s = asctime(info);
        string curTimeStr = s;
        curTimeStr.replace(curTimeStr.find("\n"), 1 , "");
        curTimeStr.replace(curTimeStr.find(":"), 1, "_");
        curTimeStr.replace(curTimeStr.find(":"), 1, "_");
        vector<string> resStr = split(curTimeStr, " ");
        string outFileName = resStr[4] + "_" + resStr[1] + "_" + resStr[2] + "_" + resStr[3] + ".txt";
//        const char* fileName = outFileName.c_str();
        string tmp = getcwd(NULL, 0);
        outFileName = tmp + "\\" + outFileName;
        ofstream outFile;
//        outFile.open(outFileName.c_str(), ios::out);
        outFile.open(outFileName.c_str(), ios::out);
        if(!outFile)
            return -1;
        while(true){

            // 接受数据
            int ret = recv(sClient, revData, 4096, 0);
            if(ret > 0){
                revData[ret] = 0x00;
                printf(revData);
            }
            string writeData = revData;
            outFile << writeData << endl;

            // 发送数据
            ret = send(sClient, sendData, strlen(sendData), 0);
            if(ret < 0)
                break;
        }
        outFile.close();
        closesocket(sClient);
    }
    WSACleanup();
    getchar();
    return 0;
}
