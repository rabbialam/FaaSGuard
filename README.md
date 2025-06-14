# FaaSGuard (Initial artifacts)

> **FaaSGuard** is a proof-of-concept framework for and training machine-learning models **(training_model)** to detect anomalies in serverless / Function-as-a-Service (FaaS).

---

## 📁 Repository Layout


- **`cpu_mem_usage/`**: Contains raw logs of CPU and memory usage during function execution.
- **`packet_capture/container_model_inference/`**: Contains the packet capture code for model inference inside containers.
- **`packet_capture/training-data-collection-image/`**: Contains the packet capture code for training phase.
- **`training_model/auto-encoder-trainer.py`**: Trains an AutoEncoder model for each function.
- **`training_model/packet_analyzer.py`**: Converts raw network packet logs into JSON files.
- **`training_model/sql_network_flow_to_json.py`**: Converts structured SQL data into JSON flow records.
