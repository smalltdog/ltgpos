#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <Python.h>
#include <torch/script.h>

#include "wavetf.h"


using std::vector;


Class WaveClf {
    public:
        WaveClf(const std::string& weight);
        ~WaveClf();

        void predict(int freq, vector<double> data);
    private:
        PyObject* tf;
}
