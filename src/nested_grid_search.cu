#include "nested_grid_search.h"


/**
 * @brief device函数，计算二维直角坐标的欧氏距离
 * @param  x1               x1
 * @param  y1               y1
 * @param  x2               x2
 * @param  y2               y2
 * @return F 计算出来的距离
 */
inline  __device__ F D_GetDistance_Cartesian2D(F x1, F y1, F x2, F y2)
{
    F dx = x1 - x2;
    F dy = y1 - y2;
    F distance = sqrt(dx * dx + dy * dy);
    return distance;
}


/**
 * @brief device函数，计算三维直角坐标的欧氏距离
 * @param  x1               x1
 * @param  y1               y1
 * @param  z1               z1
 * @param  x2               x2
 * @param  y2               y2
 * @param  z2               z2
 * @return F 计算出来的距离
 */
inline  __device__ F D_GetDistance_Cartesian3D(F x1, F y1, F z1, F x2, F y2, F z2)
{
    F dx = x1 - x2;
    F dy = y1 - y2;
    F dz = z1 - z2;
    F distance = sqrt(dx * dx + dy * dy + dz * dz);
    return distance;
}


/**
 * @brief device函数，计算二维球面坐标的欧氏距离
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @return F 计算出来的距离
 */
inline  __device__ F D_GetDistance_Sphere2D(F lat1, F lng1, F lat2, F lng2)
{
    F rad_lat_A = RAD(lat1);
    F rad_lng_A = RAD(lng1);
    F rad_lat_B = RAD(lat2);
    F rad_lng_B = RAD(lng2);

    F pA = atan(RB / RA * tan(rad_lat_A));
    F pB = atan(RB / RA * tan(rad_lat_B));

    F xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B));
    F c1 = (sin(xx) - xx) * pow(sin(pA) + sin(pB), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pA) - sin(pB), 2) / pow(sin(xx / 2), 2);
    F dr = FLATTEN / 8 * (c1 - c2);
    F distance = RA * (xx + dr);

    return distance;
}


/**
 * @brief device函数，计算二维球面坐标的欧氏距离
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @return F 计算出来的距离
 */
inline  __device__ F D_GetDistance_Sphere2D_way2(F lat1, F lng1, F lat2, F lng2)
{
    F rad_lat_A = RAD(lat1);
    F rad_lng_A = RAD(lng1);
    F rad_lat_B = RAD(lat2);
    F rad_lng_B = RAD(lng2);

    F a = rad_lat_A - rad_lat_B;
    F b = rad_lng_A - rad_lng_B;
    F dist = 2 * asin(sqrt(
        pow(sin(a / 2), 2) +
        cos(rad_lat_A) * cos(rad_lat_B) *
        pow(sin(b / 2), 2)
    ));

    dist *= 6378137;
    dist -= dist * 0.0011194;
    return dist;
}


/**
 * @brief device函数，计算三维球面坐标的欧氏距离
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  asl1             高度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @param  asl2             高度2
 * @return F 计算出来的距离
 */
inline  __device__ F D_GetDistance_Sphere3D(F lat1, F lng1, F asl1, F lat2, F lng2, F asl2)
{
    F distance = D_GetDistance_Sphere2D(lat1, lng1, lat2, lng2);
    return sqrt(distance * distance + (asl1 - asl2) * (asl1 - asl2));
}


/**
 * @brief host函数，计算二维直角坐标的欧氏距离
 * @param  x1               x1
 * @param  y1               y1
 * @param  x2               x2
 * @param  y2               y2
 * @return F 计算出来的距离
 */
inline  F H_GetDistance_Cartesian2D(F x1, F y1, F x2, F y2)
{
    F dx = x1 - x2;
    F dy = y1 - y2;
    F distance = sqrt(dx * dx + dy * dy);
    return distance;
}


/**
 * @brief host函数，计算三维直角坐标的欧氏距离
 * @param  x1               x1
 * @param  y1               y1
 * @param  z1               z1
 * @param  x2               x2
 * @param  y2               y2
 * @param  z2               z2
 * @return F 计算出来的距离
 */
inline  F H_GetDistance_Cartesian3D(F x1, F y1, F z1, F x2, F y2, F z2)
{
    F dx = x1 - x2;
    F dy = y1 - y2;
    F dz = z1 - z2;
    F distance = sqrt(dx * dx + dy * dy + dz * dz);
    return distance;
}


/**
 * @brief host函数，计算二维球坐标的欧氏距离
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @return F 计算出来的距离km
 */
inline  F H_GetDistance_Sphere2D(F lat1, F lng1, F lat2, F lng2)
{
    F rad_lat_A = RAD(lat1);
    F rad_lng_A = RAD(lng1);
    F rad_lat_B = RAD(lat2);
    F rad_lng_B = RAD(lng2);

    F pA = atan(RB / RA * tan(rad_lat_A));
    F pB = atan(RB / RA * tan(rad_lat_B));

    F xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B));
    F c1 = (sin(xx) - xx) * pow(sin(pA) + sin(pB), 2) / pow(cos(xx / 2), 2);
    F c2 = (sin(xx) + xx) * pow(sin(pA) - sin(pB), 2) / pow(sin(xx / 2), 2);
    F dr = FLATTEN / 8 * (c1 - c2);
    F distance = RA * (xx + dr);

    return distance;
}


/**
 * @brief host函数，计算三维球坐标的欧氏距离
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  asl1             高度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @param  asl2             高度2
 * @return F 计算出来的距离
 */
inline  F H_GetDistance_Sphere3D(F lat1, F lng1, F asl1, F lat2, F lng2, F asl2)
{
    F distance = H_GetDistance_Sphere2D(lat1, lng1, lat2, lng2);
    return sqrt(distance * distance + (asl1 - asl2) * (asl1 - asl2));
}


/**
 * @brief host函数，计算三维球坐标的欧氏距离,方法2
 * @param  lat1             纬度1
 * @param  lng1             经度1
 * @param  asl1             高度1
 * @param  lat2             纬度2
 * @param  lng2             经度2
 * @param  asl2             高度2
 * @return F 计算出来的距离
 */
inline  F H_GetDistance_Sphere3D_way2(F lat1, F lng1, F asl1, F lat2, F lng2, F asl2)
{
    F rad_lat_A = RAD(lat1);
    F rad_lng_A = RAD(lng1);
    F rad_lat_B = RAD(lat2);
    F rad_lng_B = RAD(lng2);

    F pA = atan(RB / RA * tan(rad_lat_A));
    F pB = atan(RB / RA * tan(rad_lat_B));
    F xx = acos(sin(pA) * sin(pB) + cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B));

    F A = RA - (RA + asl2) * cos(xx) + asl1;
    F B = (RA + asl2) * sin(xx);
    F distance = sqrt(A * A + B * B);
    return distance;
}


/**
 * @brief 三维球坐标的优度计算函数，在cuda thread中执行，这里传入的数组都应为cuda显存
 * @param  nOfsensor        sensor数量
 * @param  sensorLocs       F数组，sensor的坐标
 * @param  sensorTimesLocal sensor收到信号的时间
 * @param  gridInv          搜索粒度
 * @param  schDom           搜索范围数组
 * @param  outAns           F数组，在其中记录优度
 */
__global__ void chiSquareCal_sph3D(int nOfsensor, F* sensorLocs, F* sensorTimesLocal, F gridInv, F schDom[6], F* outAns)
{
    F error = 0, distTime, baseTime;
    /// 顺序为z++ => y++ => x++
    F xx = schDom[0] + gridInv * blockIdx.y;
    F yy = schDom[2] + gridInv * blockIdx.x;
    F zz = schDom[4] + gridInv * threadIdx.x;

    baseTime = D_GetDistance_Sphere3D(sensorLocs[0], sensorLocs[1], sensorLocs[2], xx, yy, zz) / C;
    for (int i = 0; i < nOfsensor; i++) {
        distTime = D_GetDistance_Sphere3D(sensorLocs[3 * i], sensorLocs[3 * i + 1], sensorLocs[3 * i + 2], xx, yy, zz) / C - baseTime - sensorTimesLocal[i];
        error += distTime * distTime;
    }

    error /= nOfsensor * nOfsensor;
    outAns[blockIdx.y * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x] = error;
}


/**
 * @brief 二维球坐标的优度计算函数，在cuda thread中执行，CPU调用，这里传入的数组都应该是cuda显存
 * @param  nOfsensor        sensor数量
 * @param  sensorLocs       F数组，sensor的坐标
 * @param  sensorTimesLocal sensor收到信号的时间
 * @param  gridInv          搜索粒度
 * @param  schDom           搜索范围数组
 * @param  outAns           F数组，在其中记录优度
 */
__global__ void chiSquareCal_sph2D(int nOfsensor, F* sensorLocs, F* sensorTimesLocal, F gridInv, F schDom[4], F* outAns)
{
    F error = 0, distTime, baseTime;
    /// 顺序为 y++ => x++
    F xx = schDom[0] + gridInv * blockIdx.x;
    F yy = schDom[2] + gridInv * threadIdx.x;

    /// 这里的单位是km/(km/ms) =  ms
    baseTime = D_GetDistance_Sphere2D(sensorLocs[0], sensorLocs[1], xx, yy) / C;
    for (int i = 0; i < nOfsensor; i++) {
        distTime = (D_GetDistance_Sphere2D(sensorLocs[2 * i], sensorLocs[2 * i + 1], xx, yy) / C - baseTime - sensorTimesLocal[i]);
        error += distTime * distTime * 1e6;
    }

    error /= nOfsensor - 2;
    // printf("xx=%lf, yy=%lf :  %lf\n", xx, yy, error);
    outAns[blockIdx.x * blockDim.x + threadIdx.x] = error;
}


/**
 * @brief 搜索入参检测
 * @param  nOfSensor        sensor数量
 * @param  sensorLocs       F数组，sensor的坐标
 * @param  sensorTimes      sensor收到信号的时间
 * @param  info_p           显存管理和信息管理结构
 * @param  outAns           输出结果数组
 * @return int 异常返回1，正常返回0
 */
int input_check(unsigned int* nOfSensor, F* sensorLocs, F* sensorTimes, Info_t* info_p, F outAns[5], bool is3d)
{
    unsigned int nOfSensor_local = *nOfSensor;
    if (nOfSensor_local < 3) goto RT1;

    for (int i = 0; i < nOfSensor_local; i++) {
        if (sensorTimes[i] == -1) {
            if (is3d)
                for (int k = i; k + 1 < nOfSensor_local; ++k) {
                    sensorTimes[k] = sensorTimes[k + 1];
                    sensorLocs[k * 3] = sensorLocs[k * 3 + 3];
                    sensorLocs[k * 3 + 1] = sensorLocs[k * 3 + 4];
                    sensorLocs[k * 3 + 2] = sensorLocs[k * 3 + 5];
                }
            else
                for (int k = i; k + 1 < nOfSensor_local; ++k) {
                    sensorTimes[k] = sensorTimes[k + 1];
                    sensorLocs[k * 2] = sensorLocs[k * 2 + 2];
                    sensorLocs[k * 2 + 1] = sensorLocs[k * 2 + 3];
                }
            nOfSensor_local -= 1;
            i -= 1;
        }
    }
    if (nOfSensor_local < 3) goto RT1;
    *nOfSensor = nOfSensor_local;
    return 0;
RT1:
    for (int i = 0; i < 5; ++i) outAns[i] = -1;
    return 1;
}


/**
 * @brief 将搜索结果dump到文件
 * @param  hChiOut          搜索结果数组
 * @param  gridXSize        搜索结果的x长度
 * @param  gridYSize        搜索结果的y长度
 * @param  gridZSize        搜索结果的z长度
 * @param  filename         需要dump的文件
 * @return int 成功返回0,否则返回1
 */
int dump_to_file(F* hChiOut, int gridXSize, int gridYSize, int gridZSize, const char* filename)
{
    FILE* fp = NULL;
    fp = fopen(filename, "w");
    if (fp == NULL) {
        fprintf(stderr, "fail to open file! \n");
        return 1;
    }
    for (int i = 0; i < gridXSize; i++) {
        for (int j = 0; j < gridYSize; j++) {
            for (int k = 0; k < gridZSize; k++) {
                fprintf(fp, "%f, ", hChiOut[i * gridYSize * gridZSize + j * gridZSize + k]);
            }
        }
        /// 回车安排在这里保证2d搜索时也能保存
        fprintf(fp, "\n");
    }
    printf("[dump] dump to %s\n", filename);
    fclose(fp);
    return 0;
}


/**
 * @brief 最近球坐标网格搜索函数，搜索方法主体
 * @param  nOfSensor        传感器数量
 * @param  sensorLocs       传感器的坐标，单位为经纬度，如[30.51139, 114.40250,30.51139, 114.40250,]，长度为nOfSensor * 2,如果是3d则长度为nOfSensor * 3
 * @param  sensorTimes      传感器收到雷电的时间，单位毫秒
 * @param  info_p           CUDA 内存管理和网格搜索信息结构
 * @param  outAns           输出结果的数组，内容结构为[推理雷电发生时间，维度，经度，高度，理论误差]
 * @param  is3d            是否是3d搜索，默认为false
 */
void nested_grid_search_sph(unsigned int nOfSensor, F* sensorLocs, F* sensorTimes, Info_t* info_p, F outAns[5], bool is3d)
{
    int gridXSizeFst = info_p->gridXSizeFst;
    int gridYSizeFst = info_p->gridYSizeFst;
    int gridZSizeFst = info_p->gridZSizeFst;
    int gridSizeFst = gridXSizeFst * gridYSizeFst * gridZSizeFst;
    int ANS_BYTES = gridSizeFst * sizeof(F);

    if (input_check(&nOfSensor, sensorLocs, sensorTimes, info_p, outAns, is3d)) return;

    if (is3d && gridZSizeFst < 1) {
        fprintf(stderr, "nested_grid_search(line %d): gridZSize >= 1 for 3D search\n", __LINE__);
        for (int i = 0; i < 5; ++i) outAns[i] = -1;
        return;
    }
    if (!is3d && gridZSizeFst != 1) {
        fprintf(stderr, "nested_grid_search(line %d): gridZSize != 1 for 2D search\n", __LINE__);
        for (int i = 0; i < 5; ++i) outAns[i] = -1;
        return;
    }

    F gridInvFst = info_p->gridInvFst;
    F gridInvSec = info_p->gridInvSec;

    /// 第二个之后的探测站都减去第一个探测站的时间，在计算时会认为第一个探测站的时间为0，这里不写做0是为了保留原始数据用于还原
    F* sensorTimesLocal = (F*)malloc(sizeof(F) * nOfSensor);
    memcpy(sensorTimesLocal, sensorTimes, sizeof(F) * nOfSensor);
    F tmp_time = sensorTimesLocal[0];
    for (int i = 0; i < nOfSensor; i++)
        sensorTimesLocal[i] -= tmp_time;

    /// 申请局部的cuda显存用于储存sensor的坐标和时间，因为sensor的个数可能有变
    /// 之后可以提前申请，设置一个最大数量即可
    F* dSensorLocs, * dSensorTimes;
    cudaError_t cudaStatus;

    /// 计算复制时的长度
    int _2or3D = is3d ? 3 : 2;
    do {
        cudaStatus = cudaMalloc((void**)&dSensorLocs, (nOfSensor * _2or3D) * sizeof(F));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "nested_grid_search(line %d): cudamalloc dSensorLocs failed!\n", __LINE__);
    } while (cudaStatus != cudaSuccess);
    do {
        cudaStatus = cudaMalloc((void**)&dSensorTimes, (nOfSensor) * sizeof(F));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "nested_grid_search(line %d): cudamalloc dSensorTimes failed!\n", __LINE__);
    } while (cudaStatus != cudaSuccess);

    /// 将sensor的坐标和时间从内存复制到显存中,一二级搜索都会用到
    cudaMemcpy(dSensorLocs, sensorLocs, nOfSensor * _2or3D * sizeof(F), cudaMemcpyHostToDevice);
    cudaMemcpy(dSensorTimes, sensorTimesLocal, nOfSensor * sizeof(F), cudaMemcpyHostToDevice);

    /// 取一下内存管理结构中的数个指针
    F* hChiOutFst = info_p->hChiOutFst;
    F* dChiOutFst = info_p->dChiOutFst;
    F* hSchDomFst = info_p->schDom;
    F* dSchDomFst = info_p->dSchDomFst;

    /// 3D搜索
    if (is3d) {
        dim3 grid(gridYSizeFst, gridXSizeFst), block(gridZSizeFst);
        chiSquareCal_sph3D << < grid, block >> > (nOfSensor, dSensorLocs, dSensorTimes, gridInvFst, dSchDomFst, dChiOutFst);
        cudaMemcpy(hChiOutFst, dChiOutFst, ANS_BYTES, cudaMemcpyDeviceToHost);
    }
    /// 2D搜索
    else {
        dim3 grid(gridXSizeFst), block(gridYSizeFst);
        chiSquareCal_sph2D << < grid, block >> > (nOfSensor, dSensorLocs, dSensorTimes, gridInvFst, dSchDomFst, dChiOutFst);
        cudaMemcpy(hChiOutFst, dChiOutFst, ANS_BYTES, cudaMemcpyDeviceToHost);
    }

    /// 找出最小的坐标，二级搜索从这里开始
    F minChiSquareFst = hChiOutFst[0];
    int minIndexFst = 0;
    for (int i = 1; i < gridSizeFst; ++i) {
        if (hChiOutFst[i] >= minChiSquareFst) continue;
        minChiSquareFst = hChiOutFst[i];
        minIndexFst = i;
    }
    // dump_to_file(hChiOutFst, gridXSizeFst, gridYSizeFst, gridZSizeFst, "./first.csv");

    /// 进行二级搜索
    if (DEFAULT_SECOND_SEARCH) {
        /// 计算二级搜索中心坐标
        F xx = hSchDomFst[0] + (minIndexFst / (gridYSizeFst * gridZSizeFst)) * gridInvFst;
        F yy = hSchDomFst[2] + ((minIndexFst / gridZSizeFst) % (gridYSizeFst)) * gridInvFst;
        F zz = 0;

        if (is3d) zz = hSchDomFst[4] + (minIndexFst % gridZSizeFst) * gridInvFst;
        /// 二级搜索范围
        F hSchDomSec[6] = { xx - SEC_GRID_SIZE * gridInvFst, xx + SEC_GRID_SIZE * gridInvFst,
                            yy - SEC_GRID_SIZE * gridInvFst, yy + SEC_GRID_SIZE * gridInvFst,
                            zz - SEC_GRID_SIZE * gridInvFst, zz + SEC_GRID_SIZE * gridInvFst };

        int gridXSizeSec = info_p->gridXSizeSec;
        int gridYSizeSec = info_p->gridYSizeSec;
        int gridZSizeSec = info_p->gridZSizeSec;
        int gridSizeSec = gridXSizeSec * gridYSizeSec * gridZSizeSec;
        ANS_BYTES = gridSizeSec * sizeof(F);

        /// 获取二级搜索显存等
        F* hChiOutSec = info_p->hChiOutSec;
        F* dChiOutSec = info_p->dChiOutSec;
        F* dSchDomSec = info_p->dSchDomSec;
        cudaMemcpy(dSchDomSec, hSchDomSec, 6 * sizeof(F), cudaMemcpyHostToDevice);

        /// 3D搜索
        if (is3d) {
            dim3 gridSec(gridYSizeSec, gridXSizeSec), blockSec(gridZSizeSec);
            chiSquareCal_sph3D << < gridSec, blockSec >> > (nOfSensor, dSensorLocs, dSensorTimes, gridInvSec, dSchDomSec, dChiOutSec);
            cudaMemcpy(hChiOutSec, dChiOutSec, ANS_BYTES, cudaMemcpyDeviceToHost);
        }
        /// 2D搜索
        else {
            dim3 gridSec(gridYSizeSec), blockSec(gridXSizeSec);
            chiSquareCal_sph2D << < gridSec, blockSec >> > (nOfSensor, dSensorLocs, dSensorTimes, gridInvSec, dSchDomSec, dChiOutSec);
            cudaMemcpy(hChiOutSec, dChiOutSec, ANS_BYTES, cudaMemcpyDeviceToHost);
        }

        /// 找出最小的坐标
        F minChiSquareSec = hChiOutSec[0];
        int minIndexSec = 0;
        for (int i = 1; i < gridSizeSec; i++) {
            if (hChiOutSec[i] >= minChiSquareSec) continue;
            minChiSquareSec = hChiOutSec[i];
            minIndexSec = i;
        }
        // dump_to_file(hChiOutSec, gridXSizeSec, gridYSizeSec, gridZSizeSec, "./second.csv");

        /// 根据搜索结果记录
        /// 通过二级搜索得到的结果
        outAns[1] = hSchDomSec[0] + (minIndexSec / (gridYSizeSec * gridZSizeSec)) * gridInvSec;
        outAns[2] = hSchDomSec[2] + ((minIndexSec / gridZSizeSec) % (gridYSizeSec)) * gridInvSec;
        outAns[3] = hSchDomSec[4] + (minIndexSec % gridZSizeSec) * gridInvSec;

        if (is3d)
            outAns[0] = tmp_time - (H_GetDistance_Sphere3D(sensorLocs[0], sensorLocs[1], sensorLocs[2], outAns[1], outAns[2], outAns[3]) / C);
        else
            outAns[0] = tmp_time - (H_GetDistance_Sphere2D(sensorLocs[0], sensorLocs[1], outAns[1], outAns[2]) / C);
        outAns[4] = minChiSquareSec;

        #ifdef DEBUG
        printf("[GridSearch] result: %lf,  %lf,  %lf,  %lf,  %lf\n", outAns[0], outAns[1], outAns[2], outAns[3], outAns[4]);
        #endif
    }
    /// 是通过一级搜索得到的结果
    else {
        outAns[1] = hSchDomFst[0] + (minIndexFst / (gridYSizeFst * gridZSizeFst)) * gridInvFst;
        outAns[2] = hSchDomFst[2] + ((minIndexFst / gridZSizeFst) % (gridYSizeFst)) * gridInvFst;
        outAns[3] = hSchDomFst[4] + (minIndexFst % gridZSizeFst) * gridInvFst;

        if (is3d)
            outAns[0] = tmp_time - (H_GetDistance_Sphere3D(sensorLocs[0], sensorLocs[1], sensorLocs[2], outAns[1], outAns[2], outAns[3]) / C);
        else
            outAns[0] = tmp_time - (H_GetDistance_Sphere2D(sensorLocs[0], sensorLocs[1], outAns[1], outAns[2]) / C);
        outAns[4] = minChiSquareFst;

        #ifdef DEBUG
        printf("[GridSearch] result: %lf,  %lf,  %lf,  %lf,  %lf", outAns[0], outAns[1], outAns[2], outAns[3], outAns[4]);
        #endif
    }

    /// 释放局部缓存
    cudaFree(dSensorLocs);
    cudaFree(dSensorTimes);
    free(sensorTimesLocal);
}


/**
 * @brief 构造 CUDA内存管理和网格搜索信息结构函数
 * @details 注意这个函数很多内存申请没有error
 * @param  gridInvFst       第一级搜索精度,单位度
 * @param  gridInvSec       二级搜索精度,单位度
 * @param  schDom           搜索范围，如[28, 32,100, 104，0，0], 要求是6个数，is3d为true时后两个为低到高
 * @param  is3d            是否是3d搜索,default false
 * @return Info_t* CUDA内存管理和网格搜索信息结构的指针
 */
Info_t* infoInit(F gridInvFst, F gridInvSec, F schDom[6], bool is3d, const int kMaxGridSize,
                 F* hChiOutFst, F* hChiOutSec, F* dChiOutFst, F* dChiOutSec)
{
    /// 初始化结构
    Info_t* pSystemInfo = (Info_t*)malloc(sizeof(Info_t));

    for (int i = 0; i < 6; i++)
        pSystemInfo->schDom[i] = schDom[i];

    if (!gridInvFst) gridInvFst = 0.03;
    if (!gridInvSec) gridInvFst = 0.001;

    /// 记录一二级搜索粒度
    pSystemInfo->gridInvFst = gridInvFst;
    pSystemInfo->gridInvSec = gridInvSec;

    int gridXSizeFst = (int)((schDom[1] - schDom[0]) / gridInvFst) + 1;
    int gridYSizeFst = (int)((schDom[3] - schDom[2]) / gridInvFst) + 1;
    int gridZSizeFst = !is3d ? 1:
                       (int)((schDom[5] - schDom[4]) / gridInvFst) + 1;

    #ifdef DEBUG
    printf("[InfoInit] GridSizeFst: %d, %d, %d\n", gridXSizeFst, gridYSizeFst, gridZSizeFst);
    #endif

    pSystemInfo->gridXSizeFst = gridXSizeFst;
    pSystemInfo->gridYSizeFst = gridYSizeFst;
    pSystemInfo->gridZSizeFst = gridZSizeFst;

    int gridSizeFst = gridXSizeFst * gridYSizeFst * gridZSizeFst;
    if (gridSizeFst > kMaxGridSize) return NULL;

    F* dSchDomFst;
    cudaError_t cudaStatus;
    do {
        /// 一级搜索范围, 在3d用6个数，2d用4个数
        cudaStatus = cudaMalloc((void**)&dSchDomFst, (6) * sizeof(F));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "nested_grid_search(line %d): cudamalloc dSchDomFst failed!\n", __LINE__);
    } while (cudaStatus != cudaSuccess);

    /// 申请到的显存写入结构中
    pSystemInfo->hChiOutFst = hChiOutFst;
    pSystemInfo->dChiOutFst = dChiOutFst;
    pSystemInfo->dSchDomFst = dSchDomFst;

    cudaMemcpy(dSchDomFst, pSystemInfo->schDom, 6 * sizeof(F), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
        fprintf(stderr, "nested_grid_search(line %d): cudamalloc dSchDomFst failed!\n", __LINE__);

    const int two_sides = 2;        /// do not change here, go to change SEC_GRID_SIZE
    int gridXSizeSec = (int)((SEC_GRID_SIZE * two_sides * gridInvFst) / gridInvSec) + 1;
    int gridYSizeSec = (int)((SEC_GRID_SIZE * two_sides * gridInvFst) / gridInvSec) + 1;
    int gridZSizeSec = !is3d ? 1 :
                       (int)((SEC_GRID_SIZE * two_sides * gridInvFst) / gridInvSec) + 1;

    #ifdef DEBUG
    printf("[InfoInit] GridSizeSec: %d, %d, %d\n", gridXSizeSec, gridYSizeSec, gridZSizeSec);
    #endif

    pSystemInfo->gridXSizeSec = gridXSizeSec;
    pSystemInfo->gridYSizeSec = gridYSizeSec;
    pSystemInfo->gridZSizeSec = gridZSizeSec;

    int gridSizeSec = gridXSizeSec * gridYSizeSec * gridZSizeSec;
    if (gridSizeSec > kMaxGridSize) return NULL;

    F* dSchDomSec;
    do {
        cudaStatus = cudaMalloc((void**)&dSchDomSec, (6) * sizeof(F));
        if (cudaStatus != cudaSuccess)
            fprintf(stderr, "nested_grid_search(line %d): cudamalloc dSchDomSec failed!\n", __LINE__);
    } while (cudaStatus != cudaSuccess);

    pSystemInfo->hChiOutSec = hChiOutSec;
    pSystemInfo->dChiOutSec = dChiOutSec;
    pSystemInfo->dSchDomSec = dSchDomSec;
    return pSystemInfo;
}


/**
 * @brief 释放 CUDA内存管理和网格搜索信息结构函数
 * @param  p                CUDA内存管理和网格搜索信息结构函数指针
 * @return int 释放成功返回0
 */
int infoFree(Info_t* p)
{
    // free(p->hChiOutFst);
    // free(p->hChiOutSec);
    // cudaFree(p->dChiOutFst);
    // cudaFree(p->dChiOutSec);
    cudaFree(p->dSchDomFst);
    cudaFree(p->dSchDomSec);
    free(p);
    p = NULL;
    return 0;
}
