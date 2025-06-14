#include "evaluate.h"
#include <fstream>
#include <pybind11/embed.h>
#include <iostream>
#include <vector>


namespace py = pybind11;

py::module calculate_score;
py::object calculateScore;

void initilize_evaluation(){
    
    try{
        //py::gil_scoped_acquire acquire;
        std::cout<<"loading module..\n";
        calculate_score = py::module::import("calculate_score");
        std::cout<<"module loaded"<< std::endl;
        // Get the Python function
        calculateScore = calculate_score.attr("calculateScore");

        std::cout<<"function loaded"<< std::endl;


    } catch (const std::exception &e){
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
}

float tokenize_sentence(const std::string &sentence){
        //py::scoped_interpreter guard{};
        float score = 0.0;
    try {
        //py::gil_scoped_acquire acquire;
        // Import the Python module
       
        // Define a test sentence
        //std::string sentence = "This is a test sentence for BERT scoring.";

        // Call the Python function and get the result
        py::object result_obj = calculateScore(sentence.c_str());
        //std::cout << "funciton return\n";
        //std::cout << "Python object type: " << py::str(result_obj.get_type()).cast<std::string>() << std::endl;

        score  =result_obj.cast<float>();
        // Output the result
        //std::cout << "Score for the sentence: " << score << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
    }
    return score;
}


void print_tensor(const std::vector<int64_t>& tensor) {
    std::cout << "Input Tensor: [";
    for (size_t i = 0; i < tensor.size(); ++i) {
        std::cout << tensor[i];
        if (i < tensor.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

/*int main() {
    // Load the tokenizer vocabulary and tokenize input
    std::string sentence = "Hello, how are you?";
    std::string vocab_file = "tokenizer/vocab.txt";
    std::vector<int64_t> input_ids = tokenize_sentence(sentence, vocab_file);
    torch::Tensor input_tensor = prepare_input_tensor(input_ids);

    // Load the custom model from TorchScript file
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("custom_model.pt");
        std::cout << "Model loaded successfully.\n";
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model.\n";
        return -1;
    }

    // Prepare input and run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    torch::Tensor output = model.forward(inputs).toTensor();
    
    // Display the output tensor
    std::cout << "Model output: " << output << std::endl;
    return 0;
}*/
