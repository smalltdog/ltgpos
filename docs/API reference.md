# API reference

## 调用参数

由数个 JSONObject 构成的 JSONArray，每个 JSONObject 为一个探测站点的数据。

| Keys            | IsRequired | Type   | Description               |
| --------------- | ---------- | ------ | ------------------------- |
| latitude        | true       | double | 探测站点纬度                |
| longitude       | true       | double | 探测站点经度                |
| altitude        | only 3D    | double | 探测站点海拔                |
| node            | true       | string | 主机-站点号字符串，如"81-12" |
| datetime        | true       | string | 探测到雷击的日期时间字符串，如"2020-9-21 11:31:18" |
| microsecond     | true       | uint32 | 单位 0.1 us                |
| signal_strength | true       | double | 探测雷击的信号强度           |

## 返回参数

JSONObject

| Keys             | IsRequired | Type         | Description        |
| ---------------- | ---------- | ------------ | ------------------ |
| datetime         | true       | string       | 雷击发生的日期时间字符串，如"2020-9-21 11:31:18" |
| time             | true       | uint32       | 单位 0.1 us         |
| latitude         | true       | double       | 雷击发生纬度的计算结果 |
| longitude        | true       | double       | 雷击发生经度的计算结果 |
| altitude         | only 3D    | double       | 雷击发生海拔的计算结果 |
| goodness         | true       | double       | 计算结果的优度        |
| current          | true       | double       | 电流强度的计算结果     |
| raw              | true       | JSONArray    | 输入数据信息          |
| allDist          | true       | double-array | 所有探测站到雷击发生地点的距离 (km)      |
| allDtime         | true       | double-array | 所有探测站检测雷击和雷击发生的时间差 (ms) |
| isInvolved       | true       | uint32-array | 站点是否参与定位计算，是为1，否则为0      |
| involvedNodes    | true       | string-array | 参与定位计算站点的主机-站点号字符串       |
| referNode        | true       | string       | 参考站的主机-站点号字符串，如"81-12"     |
| basicNodes       | true       | string-array | 初步定位的三个探测站主机-站点号字符串     |
| involvedSignalStrength | true | double-array | 参与计算站点的信号强度                  |
| involvedResultCurrent  | true | double-array | 参与计算站点的电流强度                  |
