# calculate_score.py
from tokenizers import Tokenizer

import numpy as np
import pickle

import time
import onnxruntime


def load_globals(filename="globals.pkl"):
    global global_min, global_max, input_dim
    filename = base_path+filename
    with open(filename, "rb") as f:
        globals_dict = pickle.load(f)
    global_min = globals_dict["global_min"]
    global_max = globals_dict["global_max"]
    input_dim = globals_dict["max_input"]
    print(f"Globals loaded from {filename}")

base_path ="/usr/src/app/"

tokenizer = Tokenizer.from_file(base_path+"bert.json")
input_dim = 0 

global_min=0
global_max =0


print("loading model...")
#state_dict = torch.load(base_path+"model_with_single_line_v3.pth", weights_only=False)
#model.load_state_dict(state_dict)  # Load the saved state_dict into the model
session = onnxruntime.InferenceSession(base_path+"autoencoder_2.onnx")

print("module loaded\nloading scaller....")
load_globals()
diff =global_max-global_min

def encode_feature(df):
    #categorical_cols =  ['path','type','seq']
    
    #print(f"global min {global_min} global max {global_max} diff {diff}")
    scaled_data = 0.01 + (df - global_min) / diff * (1.0 - 0.01)

    
    print(f"encoding done")
    return scaled_data


   
def create_sequences(data, sequence_length):
    sequences = []
    num_full_sequences = len(data) // sequence_length  # Number of full sequences
    for i in range(num_full_sequences):
        start_idx = i * sequence_length
        seq = data[start_idx:start_idx + sequence_length]
        sequences.append(seq)
    return np.array(sequences,dtype=np.float32)

def evaluate( X_test):

    reconstructed = session.run(['output'], {'input': X_test})[0]  # Assuming 'output' and 'input' are the names from export

    # Calculate Mean Squared Error element-wise without PyTorch
    error = (reconstructed - X_test) ** 2  # Element-wise squared difference

    # Compute the mean over sequence length and feature dimensions
    reconstruction_error = error.mean(axis=(1, 2))  # Mean over specified dimensions
    
    return reconstruction_error


def tokenize_with_tokenizers( sentence_list, max_length):
    encoded = tokenizer.encode_batch(sentence_list)

    # Initialize input_ids and attention_mask
    input_ids = []
    attention_mask = []

    for enc in encoded:
        # Truncate or pad input IDs to max_length
        ids = enc.ids[:max_length] + [0] * (max_length - len(enc.ids)) if len(enc.ids) < max_length else enc.ids[:max_length]
        #mask = [1] * min(len(enc.ids), max_length) + [0] * max(0, max_length - len(enc.ids))

        input_ids.append(ids)
       # attention_mask.append(mask)

    # Convert to NumPy arrays
    return {
        "input_ids": np.array(input_ids),
        #"attention_mask": np.array(attention_mask),
    }


def calculateScore(sentence: str) :
    # Tokenize input sentence
    #print("in calculate score")
    sentenceList=[]
    sentenceList.append(sentence)
    start = time.time()
    #inputs = tokenizer(sentenceList,return_tensors='np', padding='max_length', truncation=True, max_length=input_dim)
    inputs = tokenize_with_tokenizers(sentenceList,input_dim)

    data =  inputs["input_ids"]
    print(f"Tokenization time: : {time.time()-start}")

  
    start = time.time()
    arr = encode_feature(data)


    seq = create_sequences(arr,1)
    print(seq.shape)
    print(f"Encoding time: : {time.time()-start}")


    start = time.time()
    score = evaluate(seq)
    print(f"Score generation time: : {time.time()-start}")


    print("printing message")
    return score.item()

