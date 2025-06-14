//#include <torch/torch.h>
//#include <torch/script.h>
//#include <sentencepiece_processor.h>
#include <iostream>
#include <vector>
#include <string>

float tokenize_sentence(const std::string& sentence);

void print_tensor(const std::vector<int64_t>& tensor);

void initilize_evaluation();