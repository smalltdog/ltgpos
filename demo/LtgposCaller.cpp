#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jni.h>

#include "../src/ltgpos.h"
#include "LtgposCaller.h"


jstring charToJstring(JNIEnv* env, char* cstr)
{
    if (!cstr) return NULL;
    jclass strClass = (env)->FindClass("Ljava/lang/String;");
    jmethodID mid = (env)->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");

    // Convert C string to ByteArray.
    jbyteArray bytes = (env)->NewByteArray(strlen(cstr));
    (env)->SetByteArrayRegion(bytes, 0, strlen(cstr), (jbyte*) cstr);
    free(cstr);

    // Convert ByteArray to Java String.
    jstring encoding = (env)->NewStringUTF("GB2312");
    return (jstring)(env)->NewObject(strClass, mid, bytes, encoding);
}


char* jstringToChar(JNIEnv* env, jstring jstr)
{
    char* cstr = NULL;
    jclass strClass = env->FindClass("java/lang/String");
    jstring encoding = env->NewStringUTF("GB2312");
    jmethodID mid = env->GetMethodID(strClass, "getBytes", "(Ljava/lang/String;)[B");

    jbyteArray bytes = (jbyteArray) env->CallObjectMethod(jstr, mid, encoding);
    jsize bslen = env->GetArrayLength(bytes);
    jbyte* b = env->GetByteArrayElements(bytes, JNI_FALSE);

    if (bslen > 0) {
        cstr = (char*) malloc(bslen + 1);
        memcpy(cstr, b, bslen);
        cstr[bslen] = 0;
    }
    env->ReleaseByteArrayElements(bytes, b, 0);
    return cstr;
}


JNIEXPORT jint JNICALL Java_LtgposCaller_initSysInfo
  (JNIEnv* env, jobject obj) {
    return (jint)initSysInfo();
}


JNIEXPORT void JNICALL Java_LtgposCaller_freeSysInfo
  (JNIEnv* env, jobject obj) {
    freeSysInfo();
    return;
}


JNIEXPORT jstring JNICALL Java_LtgposCaller_ltgpos
  (JNIEnv* env, jobject obj, jstring jstr) {
    char* cstr = jstringToChar(env, jstr);
    char* rstr = ltgpos(cstr);
    return charToJstring(env, rstr);
}
