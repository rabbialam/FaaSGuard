# %%
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, RobertaTokenizer, RobertaModel 
from sklearn.model_selection import train_test_split
from packet_analyzer import *
import os

# %%
def json_to_sentence(data_field):
    """
    Convert a JSON object (dictionary) to a sentence where each key-value pair is represented as 'key value'.
    
    Args:
    - data_field (dict): The 'data' field of a JSON object (key-value pairs).
    
    Returns:
    - str: A single sentence where each key-value pair is represented as 'key value'.
    """
    sentence = " ".join([f"{key} {value} " for key, value in data_field.items()])
    return sentence

# %%
def parse_network_operation_line(line):

    value = line['data']                
    sentence = " ".join([f"{k} {v} " for k, v in value.items()])
    line['data'] = sentence

    return line

# %%
def parse_network_operation_multi_line(lines):
   
    #print(lines)
    record = {}
    sentence = ""
    for line in lines:
        seq = line['seq']
        for key,value in line.items():
            if key == 'data':                
                sentence = sentence.join([f"{k} {v} " for k, v in value.items()])
            elif key=='path':
                record[f'{key}_{seq}'] = value

    
    #print(record)
    record['data'] = sentence
    return record

# %%
#parse_network_operation_line('{"protocol": "HTTP", "type": "IN", "data": {"id": "4579875", "gotPrice": "true", "price": "150"}, "socket": "20", "path": "/function/product-purchase-get-price", "seq": 3}')

# %%
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

def find_max_length(sentences):

    max_length = 0
    for sentence in sentences:
        # Tokenize the sentence without padding or truncation
        tokens = tokenizer(sentence, return_tensors='pt', padding=False, truncation=False)
        # Update max_length with the largest number of tokens
        max_length = max(max_length, tokens['input_ids'].shape[1])
    return max_length

# %%
# Re-read the dataset and parse each line

def file_to_dataframe(file_path):
    data = []
    tmp_data=[]
    prev_seq =0
    with open(file_path, 'r') as file:
        for line in file:
            line_dict = json.loads(line)
            record = parse_network_operation_line(line_dict)

            if record['seq']==0 and len(tmp_data)>0:
                discart = False
                for d in tmp_data:
                    if 'failure' in d['data']:
                        discart = True
                        print("failure discurting")
                        break
                if not discart:
                    #
                    sentences=[]
                    for tmp in tmp_data:
                        sentences.append(tmp['data'])

                    sendtence_dict ={"data":" ".join(sentences)}
                    data.append(sendtence_dict)
                tmp_data=[]
            tmp_data.append(record)

# Convert list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    print(f"Data persing done the size is {df.index}")

    return df



# %%

def sentences_to_vectors(df,max_len):
    """
    Convert sentences in the 'data' column of a pandas DataFrame to vectors using a transformer model (BERT).
    Automatically determines the maximum number of tokens for padding and truncation.
    
    Args:
    - df (pd.DataFrame): DataFrame containing a 'data' column with sentences.
    
    Returns:
    - None: Modifies the 'data' column in the DataFrame in place, replacing it with vectorized embeddings.
    """
    # Find the max length based on the longest tokenized sentence in the data column
    sentences = df['data'].tolist()
    if max_len ==0:
        max_len = find_max_length(sentences)

    
    print(f"Identified max token length: {max_len}")
    
    # Store vectors in a list
    vectors = []
    
    # Iterate over each sentence in the 'data' column
    #for sentences in df['data']:
        # Tokenize the sentence and pad/truncate to the identified max_length
    inputs = tokenizer(sentences, return_tensors='np', padding='max_length', truncation=True, max_length=max_len)
        
        # Get the BERT embeddings for the input string
    #with torch.no_grad():
     #   outputs = model_bert(**inputs)
        
        # Extract the CLS token's embedding as the sentence vector
    #cls_embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        
        # Append the vector to the list
    #vectors.append(cls_embedding)
    
    # Replace the 'data' column in the original DataFrame with the vectorized data
    return inputs['input_ids'],max_len
    

# %%
scaler = MinMaxScaler(feature_range=(0.01, 1))  # Scaling from 0.01 to 1
global_min=0
global_max =0
diff =0

def fit_scaler(data):
    global global_min, global_max, diff

    print("fit scaller called")
  
    non_zero_data = data[data > 0]
    global_min = non_zero_data.min()
    global_max = data.max()
    diff = global_max-global_min




label_encoders = {}

def encode_feature(df):
  
    print(f"global min {global_min} global max {global_max} diff {diff}") # see global min max from bert token list
    scaled_data = 0.01 + (df - global_min) / diff * (1.0 - 0.01)

    
    print(f"encoding done")
    return scaled_data


import numpy as np

def create_sequences(data, sequence_length):
    sequences = []
    num_full_sequences = len(data) // sequence_length  # Number of full sequences
    for i in range(num_full_sequences):
        start_idx = i * sequence_length
        seq = data[start_idx:start_idx + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

# %%
def prepare_data_from_file(file_path,max_len=0,fit=True):
    df = file_to_dataframe(file_path)
  
    data_array,max_len = sentences_to_vectors(df,max_len)
   
    if fit:
        fit_scaler(data_array)
    data_array = encode_feature(data_array)

    sequence_length = 1

    sequences = create_sequences(data_array, sequence_length)

    print(len(sequences))
    return sequences,max_len


class GRU_Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.2):
        super(GRU_Autoencoder, self).__init__()
        
        self.hidden_dim = hidden_dim

        # Encoder GRU
        self.encoder_gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout)

        # Fully connected layer to map GRU hidden state to latent space
        self.fc_enc = nn.Linear(hidden_dim, latent_dim)

        # Fully connected layer to map latent space back to GRU hidden state
        self.fc_dec = nn.Linear(latent_dim, hidden_dim)

        # Decoder GRU (hidden_dim -> hidden_dim)
        self.decoder_gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout)

        # Linear layer to map hidden_dim to input_dim
        self.fc_out = nn.Linear(hidden_dim, input_dim)

        # Activation function (optional)
        self.activation = nn.Sigmoid()  # Use based on the data type

    def forward(self, x):
        # Encode the input sequence
        enc_out, enc_hidden = self.encoder_gru(x)  # enc_hidden: (num_layers, batch_size, hidden_dim)
        enc_hidden = enc_hidden[-1]  # Get the last hidden state (batch_size, hidden_dim)

        # Map the hidden state to the latent space
        latent = self.fc_enc(enc_hidden)  # (batch_size, latent_dim)

        # Map the latent space back to the GRU hidden state
        dec_hidden = self.fc_dec(latent).unsqueeze(0)  # (1, batch_size, hidden_dim)
        dec_hidden = dec_hidden.repeat(self.decoder_gru.num_layers, 1, 1)  # Repeat for GRU layers

        # Initialize decoder input (zeros)
        decoder_input = torch.zeros(x.size(0), x.size(1), self.hidden_dim).to(x.device)  # Initialize decoder input

        # Decode the latent representation
        dec_out, _ = self.decoder_gru(decoder_input, dec_hidden)  # dec_out: (batch_size, seq_length, hidden_dim)

        # Map from hidden_dim back to input_dim
        dec_out = self.fc_out(dec_out)  # (batch_size, seq_length, input_dim)

        # Apply activation (if needed)
        dec_out = self.activation(dec_out)  # Optional, based on your data
        return dec_out


import torch.utils.data as dat
def train_model(model, X_train, num_epochs, batch_size,criterion,optimizer):
    model.train()
    loader = dat.DataLoader(dat.TensorDataset(X_train), shuffle=True, batch_size=batch_size)
    losses=[]
    for epoch in range(num_epochs):

        
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = X_train[indices]

            outputs = model(batch_x)
            #print(outputs.shape)
            loss = criterion(outputs, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= X_train.size(0)
        losses.append(epoch_loss)
        if epoch%20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
    
    return losses


def evaluate(model, X_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        reconstructed = model(X_test)  # Forward pass
        mse_loss = nn.MSELoss(reduction='none')  # Element-wise loss
        error = mse_loss(reconstructed, X_test)  # Shape: (batch_size, seq_length, num_features)
        
        # Compute the mean over the sequence length and feature dimensions
        print(error.shape)
        reconstruction_error = error.mean(dim=[ 1,2])  # Shape: (batch_size,)
    return reconstruction_error



# %%
import matplotlib.pyplot as plt
def main():
    directory = 'training_data_directory'
    output_dir = 'output_directory'
    # file_path = 'output_data_small.txt'
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"Processing file: {file_path}")
            process_container(output_dir, file_path)

def process_container(output_dir, file_path):
    base_name = os.path.basename(file_path)  # Returns 'file_name.txt'
    dir_name = os.path.splitext(base_name)[0]
    full_path = os.path.join(output_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)

    print(f">>>>>>>>>>>>>>>>Training for funciton {dir_name} <<<<<<<<<<<<<<<<<<<<<<")
    
    sequences,max_length = prepare_data_from_file(file_path)
    train_size = int(len(sequences) * 0.8)
    X_train_seq = sequences[:train_size]
    X_test_seq = sequences[train_size:]

# Convert to PyTorch tensors
    X_train = torch.tensor(X_train_seq, dtype=torch.float32)
    X_test = torch.tensor(X_test_seq, dtype=torch.float32)
    print(X_train.shape)

    #X_train.shape[1]
    X_train.shape[2]

# Initialize the model, loss function, and optimizer
    input_dim = X_train.shape[2] #second dim is the main data / If modified change it to 1
    hidden_dim = 256
    latent_dim = 64
    num_layers = 2
    model = GRU_Autoencoder(input_dim, hidden_dim, latent_dim, num_layers) #AE(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
    num_epochs = 200
    batch_size = 20

    print("Training begain")
   
    losses=train_model(model, X_train, num_epochs, batch_size,criterion,optimizer)

    filename = os.path.join(full_path, 'Epoch vs Training.png')
    plt.plot(range(1, num_epochs + 1), losses)
    plt.title('Epoch vs Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    #plt.show()
    plt.savefig(filename, dpi=300, bbox_inches='tight')


    threashold=generate_error_threashold(model,X_test,full_path)

    save_globals(model,max_length,threashold,full_path)

def generate_error_threashold(model,X_test,full_path):
    reconstruction_error = evaluate(model, X_test)
    function = os.path.basename(full_path)
# Print the reconstruction error for each sequence
    print("Reconstruction Error per Sequence:")
    #print(reconstruction_error)

    print(reconstruction_error.numpy())
    np.savetxt(os.path.join(full_path, f'{function}_Reconstruction_error.txt'), reconstruction_error.numpy(),  fmt='%.6e')
    filename = os.path.join(full_path, f'{function}_Reconstruction_error.pdf')
    plt.figure()
    plt.hist(reconstruction_error.numpy(), bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Number of packets')
    plt.title(f'{function} Error Distribution')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    #plt.show()

# Define a threshold (e.g., mean + 3 standard deviations)
    threshold = reconstruction_error.mean() + 3 * reconstruction_error.std() 
# Identify anomalies
    anomalies = reconstruction_error > threshold
    print(threshold)
# Print indices of anomalous sequences
    anomalous_indices = torch.where(anomalies)[0]
    print(f"Total vectors {len(reconstruction_error)} and annomalies {len(anomalous_indices)}")
    print("Anomalous Sequences at Indices:", anomalous_indices)
    return threshold
# investigate annomalies

def save_model_as_onnx(model,max_length,full_path):
    model.eval()

# Create a dummy input that has the same dimensions as the model's input
    dummy_input = torch.randn(1,1, max_length)  # Replace `input_size` with the appropriate dimensions
    filename = os.path.join(full_path, 'autoencoder_2.onnx')
# Export the model
    torch.onnx.export(
        model,               # the model being converted
        dummy_input,         # the dummy input for tracing
        filename,  # the output file name
        export_params=True,  # store the trained parameter weights inside the model
        opset_version=11,    # the ONNX version to export to
        do_constant_folding=True,  # optimize constant expressions
        input_names=['input'],     # input name in the model
        output_names=['output']    # output name in the model
    )
    

# %%
import pickle
from tokenizers import Tokenizer

def save_globals(model,max_length,threshold,full_path):
    global global_min, global_max, diff
    filename = os.path.join(full_path, 'globals.pkl')
    globals_dict = {"global_min": global_min, "global_max": global_max, "max_input": max_length,"threshold":threshold}
    with open(filename, "wb") as f:
        pickle.dump(globals_dict, f)
    print(f"Globals saved to {filename}")
    save_model_as_onnx(model,max_length,full_path)
    print("Model seved") 



if __name__ == "__main__":
    main()


'''

'''



