#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include <Python.h>
#include <torch/script.h>

#include "wavetf.h"


template<typename T>
static PyObject* vec2tuple(const vector<T>& vec, int type = PyArray_FLOAT)
{
    PyObject* tuple = PyTuple_New(vector.size());
    for (int i = 0; i < vec.size(); i++) {
        PyTuple_SetItem(tuple, i, PyFloat_FromDouble(vec[i]));
    }
    return tuple;
}


Class WaveClf : public torch::jit::script::Module
{
    public:
        WaveClf(const std::string& weight);
        ~WaveClf();

        int forward(std::vector<torch::jit::IValue>);
        void build_transformer();
    private:
        PyObject* tf;
}


WaveClf::WaveClf(const std::string& weight)
{
    try {
        *this = torch::jit::load(weight);
    }
    catch (const c10::Error& e) {
        std::cerr << __FILE__ << __LINE__ << ": " << "error loading the model.\n";
    }

    PyImport_AppendInittab("wavetf", PyInit_wavetf);
    Py_Initialize();
    PyImport_ImportModule("wavetf");
}


int WaveClf::forward(int freq, vector<double> data)
{
    input = wavetf(freq, vec2tuple(data), this->tf);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    at::Tensor output = this->forward(inputs).toTensor();
    return output.slice(/*dim=*/1, /*start=*/0, /*end=*/5);
}


void build_transformer() {
    this->tf = ;
}


WaveClf::~WaveClf() {
    Py_Finalize();
}
