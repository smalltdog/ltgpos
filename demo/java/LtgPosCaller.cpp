#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jni.h>

#include "../../src/lightning_position.h"
#include "LtgPosCaller.h"


jstring charTojstring(JNIEnv* env, const char* cstr)
{
    if (!cstr) return NULL;         // TODO return error
    jclass strClass = (env)->FindClass("Ljava/lang/String;");
    // 获取 String(byte[], String) 的构造器，用于将本地 byte 数组转换为一个新 String
    jmethodID ctorID = (env)->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    // 建立 byte 数组
    jbyteArray bytes = (env)->NewByteArray(strlen(cstr));
    // 将 char* 转换为 byte 数组
    (env)->SetByteArrayRegion(bytes, 0, strlen(cstr), (jbyte*) cstr);
    // 设置 String，保存语言类型，用于 byte 数组转换至 String 时的参数
    jstring encoding = (env)->NewStringUTF("GB2312");
    // 将 byte 数组转换为 java String，并输出
    return (jstring)(env)->NewObject(strClass, ctorID, bytes, encoding);
}


char* jstringToChar(JNIEnv* env, jstring jstr)
{
    char* rtn = NULL;
    jclass clsstring = env->FindClass("java/lang/String");
    jstring strencode = env->NewStringUTF("GB2312");
    jmethodID mid = env->GetMethodID(clsstring, "getBytes", "(Ljava/lang/String;)[B");
    jbyteArray barr = (jbyteArray) env->CallObjectMethod(jstr, mid, strencode);
    jsize alen = env->GetArrayLength(barr);
    jbyte* ba = env->GetByteArrayElements(barr, JNI_FALSE);
    if (alen > 0) {
        rtn = (char*) malloc(alen + 1);
        memcpy(rtn, ba, alen);
        rtn[alen] = 0;
    }
    env->ReleaseByteArrayElements(barr, ba, 0);
    return rtn;
}


JNIEXPORT jstring JNICALL Java_LtgPosCaller_ltgPosition(JNIEnv* env, jobject obj, jstring j_str)
{
    char* c_str = jstringToChar(env, j_str);
    char* result = ltgPosition(c_str);
    return charTojstring(env, result);
}


JNIEXPORT int JNICALL Java_LtgPosCaller_mallocResBytes(JNIEnv* env, jobject obj)
{
    return (jint)mallocResBytes();
}


JNIEXPORT void JNICALL Java_LtgPosCaller_freeResBytes(JNIEnv* env, jobject obj)
{
    freeResBytes();
    return;
}


JNIEXPORT void JNICALL Java_LtgPosCaller_setCfg(JNIEnv* env, jobject obj, jint maxNumSensors,
    jint maxGridSize, jdouble schDomRatio, jdouble dtimeThreshold, jboolean isInvCal)
{
    setCfg(maxNumSensors, maxGridSize, schDomRatio, dtimeThreshold, isInvCal);
    return;
}


JNIEXPORT void JNICALL Java_LtgPosCaller_setCfgFromFile(JNIEnv* env, jobject obj, jstring j_str)
{
    char* c_str = jstringToChar(env, j_str);
    setCfgFromFile(c_str);
    return;
}
