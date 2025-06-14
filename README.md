# FaaSGuard (Initial artifacts)

> **FaaSGuard** is a proof-of-concept framework for and training machine-learning models **(training_model)** to detect anomalies in serverless / Function-as-a-Service (FaaS).

---

## 📁 Repository Layout

FaaSGuard/
|__ cpu_mem_usage (Contains cpu and memory usage raw data)
├── packet_capture/
│ ├── container_model_inferance/ (This directory contains the code related to model inferance)
│ └── training-data-collection-image/ (This directory contains the base docker image used for training data collection)
└── training_model/
├── auto-encoder-trainer.py (Uses the training data to train the model and store model and related data)
├── packet_analyzer.py (This script processes the raw data from packet capture)
└── sql_network_flow_to_json.py