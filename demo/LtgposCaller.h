/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class LtgposCaller */

#ifndef _Included_LtgposCaller
#define _Included_LtgposCaller
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     LtgposCaller
 * Method:    initSysInfo
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_LtgposCaller_initSysInfo
  (JNIEnv *, jobject);

/*
 * Class:     LtgposCaller
 * Method:    freeSysInfo
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_LtgposCaller_freeSysInfo
  (JNIEnv *, jobject);

/*
 * Class:     LtgposCaller
 * Method:    ltgpos
 * Signature: (Ljava/lang/String;)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_LtgposCaller_ltgpos
  (JNIEnv *, jobject, jstring);

#ifdef __cplusplus
}
#endif
#endif
