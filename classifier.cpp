#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "torch/script.h"
#include "preprocessing.h"


Class WaveClf : public torch::jit::script::Module {
    public:
        WaveClf(const std::string& weight);
        forward(std::vector<torch::jit::IValue>);
}


WaveClf::WaveClf(const std::string& weight)
{
    try {
        *this = torch::jit::load(weight);
    }
    catch (const c10::Error& e) {
        std::cerr << __FILE__ << __LINE__ << ": " << "error loading the model\n";
    }
}


WaveClf::forward(int freq, vector<double> input)
{
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::zeros({1, 3, 9, 112, 112}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = this->forward(inputs).toTensor();
    return output.slice(/*dim=*/1, /*start=*/0, /*end=*/5);
}