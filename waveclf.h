#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <Python.h>
#include <torch/script.h>

#include "wavetf.h"


using std::vector;


class WaveClf {
    public:
        WaveClf(const std::string& weight);
        ~WaveClf();

        void predict(int freq, vector<double> data);
    private:
        torch::jit::Module module;
        PyObject* tf;
};
