#include "waveclf.h"


template <typename T>
PyObject* vec2tuple(const vector<T>& vec)
{
    PyObject* tuple = PyTuple_New(vec.size());
    for (int i = 0; i < vec.size(); i++) {
        PyTuple_SetItem(tuple, i, PyFloat_FromDouble(vec[i]));
    }
    return tuple;
}


template <typename T>
void list2vec(PyObject* list, vector<T>& vec)
{
    for (int i = 0; i < PyList_Size(list); i++) {
        vec.push_back(PyFloat_AsDouble(PyList_GetItem(list, i)));
    }
    return;
}


WaveClf::WaveClf(const std::string& weight)
{
    try {
        this->module = torch::jit::load(weight);
    }
    catch (const c10::Error& e) {
        std::cerr << __FILE__ << __LINE__ << ": " << "error loading the model.\n";
    }

    PyImport_AppendInittab("wavetf", PyInit_wavetf);
    Py_Initialize();
    PyImport_ImportModule("wavetf");

    this->tf = build_tf();
}


void WaveClf::predict(int freq, vector<double> data)
{
    PyObject* input_1 = vec2tuple(data);
    PyObject* input_2 = wavetf(freq, input_1, this->tf);\
    Py_DECREF(input_1);

    vector<double> input;
    list2vec(input_2, input);
    Py_DECREF(input_2);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor(input));

    at::Tensor output = this->module.forward(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
    return;
}


WaveClf::~WaveClf() {
    Py_Finalize();
}
