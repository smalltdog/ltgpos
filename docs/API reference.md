# API reference

## 调用参数

由数个 JSONObject 构成的 JSONArray，每个 JSONObject 为一个探测站点的数据。

| Keys            | Type   | Description               |
| --------------- | ------ | ------------------------- |
| latitude        | double | 探测站点纬度                |
| longitude       | double | 探测站点经度                |
| node            | string | 主机-站点号字符串，如"81-12" |
| datetime        | string | 探测到雷击的日期时间字符串，如"2020-9-21 11:31:18" |
| microsecond     | uint32 | 单位 0.1 us                |
| signal_strength | double | 探测雷击的信号强度           |

## 返回参数

JSONObject

| Keys                | Type         | Description        |
| ------------------- | ------------ | ------------------ |
| datetime            | string       | 雷击发生的日期时间字符串，如"2020-9-21 11:31:18" |
| time                | uint32       | 单位 0.1 us         |
| latitude            | double       | 雷击发生纬度的计算结果 |
| longitude           | double       | 雷击发生经度的计算结果 |
| altitude            | double       | 雷击发生海拔的计算结果 |
| goodness            | double       | 计算结果的优度        |
| current             | double       | 雷击电流的计算结果     |
| raw                 | JSONArray    | 原始输入数据          |
| allDist             | double-array | 所有探测站点与雷击的距离，单位 km      |
| allDtime            | double-array | 所有探测时间与雷击的时差，单位 ms      |
| isInvolved          | uint32-array | 所有站点是否参与定位计算，是为1，否则为0 |
| involvedNodes       | string-array | 参与定位计算站点的主机-站点号字符串     |
| referNode           | string       | 参考站的主机-站点号字符串，如"81-12"   | 
| involvedSigStrength | double-array | 参与计算站点的信号强度 |
| involvedCurrent     | double-array | 参与计算站点的电流强度 |
