#!/bin/bash

# Define the namespace to list pods from
NAMESPACE="openfaas-fn"
# Directory on local machine to save files
LOCAL_DIR="data_store_path"
# File to store all non-JSON hex strings
HEX_STRING_FILE="$LOCAL_DIR/non_json_hex_strings.txt"

# Create directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# List all pods in the namespace
pods=$(kubectl get pods -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}')

# Loop through each pod
for pod in $pods; do
    # Define the directory to store files from this pod
	echo "coping from $pod"
	
    POD_DIR="$LOCAL_DIR/$pod"
    mkdir -p "$POD_DIR"

    # Copy files from /tmp in the pod
    kubectl cp "$NAMESPACE/$pod:/tmp/" "$POD_DIR" 

    # Go through each file copied from the pod
    
done

echo "Files copied and hex strings extracted."
