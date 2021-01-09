#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jni.h>

#include "../src/ltgpos.h"
#include "LtgposCaller.h"


jstring cstr2jstr(JNIEnv* env, char* cstr)
{
    if (!cstr) return NULL;
    jclass Jstr = env->FindClass("Ljava/lang/String;");
    jmethodID mid = env->GetMethodID(Jstr, "<init>", "([BLjava/lang/String;)V");

    // Convert C string to ByteArray.
    jbyteArray bytes = env->NewByteArray(strlen(cstr));
    env->SetByteArrayRegion(bytes, 0, strlen(cstr), (jbyte*) cstr);

    // Convert ByteArray to Java String.
    jstring enc = env->NewStringUTF("GB2312");
    return (jstring)env->NewObject(Jstr, mid, bytes, enc);
}


char* jstr2cstr(JNIEnv* env, jstring jstr)
{
    char* cstr = NULL;
    jclass Jstr = env->FindClass("java/lang/String");
    jmethodID mid = env->GetMethodID(Jstr, "getBytes", "(Ljava/lang/String;)[B");
    jstring enc = env->NewStringUTF("GB2312");
    jbyteArray bytes = (jbyteArray) env->CallObjectMethod(jstr, mid, enc);

    jsize len = env->GetArrayLength(bytes);
    jbyte* byte = env->GetByteArrayElements(bytes, JNI_FALSE);

    if (len > 0) {
        cstr = (char*) malloc(len + 1);
        memcpy(cstr, byte, len);
        cstr[len] = 0;
    }
    env->ReleaseByteArrayElements(bytes, byte, 0);
    return cstr;
}


JNIEXPORT void JNICALL Java_LtgposCaller_freeSysInfo(JNIEnv* env, jobject obj)
{
    freeSysInfo();
    return;
}


JNIEXPORT jstring JNICALL Java_LtgposCaller_ltgpos(JNIEnv* env, jobject obj, jstring jstr)
{
    char* cstr = jstr2cstr(env, jstr);
    char* rstr = ltgpos(cstr);
    free(cstr);
    return cstr2jstr(env, rstr);
}
