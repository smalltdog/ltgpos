class LtgPosCaller {
    // 并行雷电定位计算
    public native String ltgPosition(String str);

    // 为网格搜索计算结果分配 Host 和 Device 内存空间
    public native int mallocResBytes();
    // 释放为网格搜索计算结果分配的 Host 和 Device 内存空间
    public native void freeResBytes();

    /**
     * @brief 为系统计算设定配置参数
     * @param  maxNumSensors    最大检测站点数，默认 64
     * @param  maxGridSize      最大搜索网格数，默认 80 * 80 * 80
     * @param  schDomRatio      搜索区域扩大比例，默认 1.2
     * @param  dtimeThreshold   反演时选取阈值，默认 1 km / C km/ms
     * @param  isInvCal         是否进行初筛以及反演计算，默认 true
     * @return Info_t* CUDA内存管理和网格搜索信息结构的指针
     */
    public native void setCfg(int maxNumSensors, int maxGridSize, double schDomRatio, double dtimeThreshold, boolean isInvCal);
    // 从文件中读取系统计算配置参数
    public native void setCfgFromFile(String filename);

    public static void main(String[] args) {
        LtgPosCaller instance = new LtgPosCaller();

        int isMallocSuccess = instance.mallocResBytes();
        if (isMallocSuccess == 0)                   // error handling
            System.out.println("[Malloc] Result memory malloc failed\n");

        else {
            String result = "";
            long start = System.nanoTime();

            // for(int i = 0; i < 10000; ++i)
            result = instance.ltgPosition("[ {\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8493, \"latitude\": 29.5466, \"microsecond\": 2330000, \"node\": \"waibiwaibi\", \"signal_strength\":1},{\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.7540, \"latitude\": 29.4369, \"microsecond\": 8020000, \"node\": \"waibiwaibi\", \"signal_strength\":1}, {\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8379, \"latitude\": 29.5471, \"microsecond\": 8700000, \"node\": \"waibiwaibi\", \"signal_strength\":1}, {\"datetime\": \"2020-11-23 01:01:12\", \"longitude\": 119.8429, \"latitude\": 29.5629, \"microsecond\": 9590000, \"node\": \"waibiwaibi\", \"signal_strength\":1} ]");

            System.out.println(result);
            // long end = System.nanoTime();
            // long runTime = end - start;
            // System.out.println(runTime);
        }
    }

    static {
        // System.loadLibrary("lightning_position");
        System.load("/home/jhy/repos/high-parallel-lightning-positioning/libs/liblightning_position.so");
    }

    protected void finalize() {
        freeResBytes();
    }
}