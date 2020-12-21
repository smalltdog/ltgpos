class LtgposCaller {
    public native int initSysInfo();
    public native void freeSysInfo();

    public native String ltgpos(String str);

    public static void main(String[] args) {
        LtgposCaller instance = new LtgposCaller();



        instance.freeSysInfo();
    }

    static {
        System.load("/home/jhy/ltgpos/libs/liblitpos.so");
        // System.loadLibrary("ltgpos");
    }

    protected void finalize() {
        freeSysInfo();
    }
}
