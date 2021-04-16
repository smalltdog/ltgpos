# Doc

## Ltgpos 雷电定位

### ltgpos.cu

`ltgpos()`
- 声明：`char* ltgpos(char* str)`
- 描述：模块调用入口
- 参数：输入的数据JSON字符串
- 返回：输出的结果JSON字符串

### json_parser.cu

`parseJsonStr()`
- 声明：`cJSON* parseJsonStr(const char* jstr, schdata_t* schdata)`
- 描述：解析输入的数据JSON字符串，选取定位计算所需的数据
- 参数：输入的数据JSON字符串，供解析填充的计算数据
- 返回：输入的数据JSON字符串中的源数据字段

`formatRetJsonStr()`
- 声明：`char* formatRetJsonStr(schdata_t* schdata, cJSON* jarr)`
- 描述：根据定位计算结果，构建输出的结果JSON字符串
- 参数：定位计算结果，输入的数据JSON字符串中的源数据字段
- 返回：输出的结果JSON字符串

### grid_search.cu

`grid_search()`
- 声明：`void grid_search(ssrinfo_t* ssrinfo, grdinfo_t* grdinfo, schdata_t* schdata)`
- 描述：通过网格搜索计算定位结果
- 参数：全局计算信息，定位计算数据
- 返回：定位计算结果

`calGirdGoodness2d_G()`
- 声明：`__global__ void calGirdGoodness2d_G(ssrinfo_t sinfo, grdinfo_t ginfo)`
- 描述：网格搜索的单格点计算优度
- 参数：格点计算信息
- 返回：格点计算结果

### comb_mapper.cu

`comb_mapper()`
- 声明：`std::vector<long> comb_mapper(long involved)`
- 描述：对选取站点进行排列组合，生成多组不同的站点选取组合
- 参数：站点选取掩码
- 返回：多组站点选取掩码

### geodistance.cu

`getGeoDistance2d()`
- 声明：`double getGeoDistance2d(double lat1, double lon1, double lat2, double lon2)`
- 描述：计算两点距离（半正弦公式）
- 参数：站点经纬度
- 返回：距离（km）

### config.h

`kMaxGrdSize = 1024`：GTX 1080Ti 所支持的最大线程数
`kGoodThres = 20`：优度选取阈值

## Ltgwave 波形分类

### model

网络结构：Conv3d -> ResNet -> LSTM -> Conv2d
- Conv3d：ConvFrontend.py
- ResNet: ResNetBBC.py
- LSTM: LSTMBackend.py
- Conv2d: ConvBackend.py

### dataloader

数据集加载与预处理

### configs

`self.numclasses = 2`：指定分类类别数
