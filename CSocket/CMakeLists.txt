cmake_minimum_required(VERSION 3.19)
project(CSocket)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_EXE_LINKER_FLAGS "-static")
set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
add_executable(CSocket main.cpp)
link_libraries(ws2_32)
target_link_libraries(CSocket libwsock32.dll)