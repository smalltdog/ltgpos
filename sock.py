import numpy as np
import pandas as pd
import datetime
import socket
import os
import time

import msg_pb2


def get_msg(data=None, type='cal'):
    msg_ = msg_pb2.msg_to_server()
    msg_.msgIndex = 0

    if type == 'init':
        msg_.msgType = msg_pb2.msg_to_server.INIT
        _msg = msg_pb2.msg_to_server.INIT_msg()
        _msg.n_of_sensor = data.shape[0]
        _msg.is_3D = 0
        _msg.first_diff = 0.05
        _msg.second_diff=0.001

        _max = np.max(data, axis=0)
        _min = np.min(data, axis=0)
        max_ = _max * 1.5 - _min * 0.5
        min_ = _min * 1.5 - _max * 0.5

        _msg.searchDom.extend([min_[0], max_[0], min_[1], max_[1], 0, 0])
        _msg.sensorLoc.extend(data.flatten().tolist())
        msg_._init.CopyFrom(_msg)

    elif type == 'cal':
        msg_.msgType = msg_pb2.msg_to_server.CAL
        _msg = msg_pb2.msg_to_server.CAL_msg()
        _msg.sensorTime.extend(list(data))  # data is a tuple of time
        msg_.cal.CopyFrom(_msg)

    elif type == 'end':
        msg_.msgType = msg_pb2.msg_to_server.END
        _msg = msg_pb2.msg_to_server.END_msg()
        _msg.is_END = 1
        msg_.end.CopyFrom(_msg)

    else:
        raise TypeError('unknown message type')

    while(1):
        msgToString = msg_.SerializePartialToString()
        msg_.size = len(msgToString)
        msgToString = msg_.SerializePartialToString()
        if(len(msgToString)==msg_.size):
            break
    # print(msg_)
    return msg_, msgToString


def cuda_init(host, port, local_sock, init_msg, init_msgToString):
    """
    初始化cuda进程
    """
    local_sock.connect((host, port))
    while(1):
        print("SEND INIT MSG TO {}:port:{}".format(host, port))
        local_sock.send(init_msgToString)
        r = local_sock.recv(1024)
        rec_msg = msg_pb2.search_result_to_client()
        rec_msg.ParseFromString(r)
        if(rec_msg.msgIndex == init_msg.msgIndex and rec_msg.size ==0):
            print("SUCCESS INIT {}:port:{}".format(host, port))
            return 1
        else:
            # 返回的index和发出的不同，error
            # TODO 异常处理
            return 0


def cuda_cal(local_sock, cal_msg, cal_msgToString):
    """
    执行计算，并返回计算结果包
    """
    local_sock.send(cal_msgToString)
    r = local_sock.recv(1024)
    rec_msg = msg_pb2.search_result_to_client()
    rec_msg.ParseFromString(r)
    if(rec_msg.msgIndex == cal_msg.msgIndex and rec_msg.size ==0):
        print("SUCCESS RECV")
    else:
        # 返回的index和发出的不同，error
        # TODO 异常处理
        return 0
    print(rec_msg)
    return rec_msg


def cuda_end(local_sock, end_msg, end_msgToString):
    """
    结束cuda进程，释放资源
    """
    local_sock.send(end_msgToString)
