# Distributed Retrieval-Augmented Generation

## Prerequisites

- Ubuntu 22.04
- Python 3.10
- Ollama 0.5.7

Install Python requirements:

```bash
$ pip install -r requirements.txt 
```

Install and start [Ollama](https://ollama.com/download/linux) local LLM service:

```bash
# start ollama service:
$ sudo systemctl start ollama
# start serving model "Llama 3.2 - 3B"
# for other Ollama models: https://ollama.com/library
$ ollama run llama3.2:3b
# preload a model into Ollama to get faster response times:
$ curl http://localhost:11434/api/generate -d '{"model": "llama3.2:3b"}'
# check ollama log
$ journalctl -u ollama
```

If you do not have root permission, install and start Ollama [manually](https://github.com/ollama/ollama/blob/main/docs/linux.md):

```bash
# Download and extract the package:
$ curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz
$ sudo tar -C <your-path>/ -xzf ollama-linux-amd64.tgz
# Create user-mode systemd.service file `~/.config/systemd/user/ollama.service` with the following content:
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=<your-path>/bin/ollama serve
Restart=always
RestartSec=3
Environment="PATH=$PATH"

[Install]
WantedBy=default.target
# Start systemd ollama service in user mode:
$ systemctl --user enable ollama
$ systemctl --user start ollama
# Check ollama service log:
$ journalctl --user -f -u ollama
```

Install [NLTK Data](https://www.nltk.org/data.html)

```bash
$ python -m nltk.downloader all
```

## Models

The following Ollama models are utilized in our experiments.

- [Llama 3.2 3B](https://ollama.com/library/llama3.2:3b)
- [Gemma 2 2B](https://ollama.com/library/gemma2:2b)
- [Qwen 2.5 3B](https://ollama.com/library/qwen2.5:3b)

## Datasets

The following Hugging Face datasets are utilized in our experiments.

- [MMLU](https://huggingface.co/datasets/cais/mmlu)
- [medical_extended](https://huggingface.co/datasets/sarus-tech/medical_extended)
- [news-category-dataset](https://huggingface.co/datasets/heegyu/news-category-dataset)

> Use `HF_HUB_CACHE` to configure where repositories from the Hub will be cached locally (models, datasets and spaces).

## Run

```bash
$ python simulator.py
```

## Membership Inference Attack (MIA)

### Overview

Membership Inference Attack (MIA) is a privacy attack applied to Distributed RAG (Retrieval-Augmented Generation) systems. The primary goal of this attack is to determine whether a specific data point exists in the knowledge base. The attacker analyzes retrieval patterns and response confidence to infer membership status.

### Attack Methodology

The Membership Inference Attack follows these steps:

1. **Data Splitting**: All data points are divided into two groups:
   - **Members**: Data points that exist in the knowledge base
   - **Non-members**: Data points that are temporarily removed from the knowledge base

2. **Feature Extraction**: For each data point, the following features are collected:
   - **Semantic Similarity**: Semantic similarity between the RAG system's answer and the original answer (60% weight)
   - **Normalized Hops**: Number of network hops required for query resolution (30% weight)
   - **Answer Length Ratio**: Ratio of RAG answer length to original answer length (10% weight)

3. **Membership Score**: A membership score is calculated using the above features:
   ```
   membership_score = (semantic_similarity × 0.6) + 
                      ((1 - normalized_hops) × 0.3) + 
                      (min(answer_length_ratio, 1.0) × 0.1)
   ```

4. **Threshold-based Classification**: Membership prediction is performed using a threshold percentile.

5. **Metrics Calculation**: Various metrics are calculated to evaluate attack success:
   - Attack Accuracy
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - Precision
   - Recall
   - AUC-ROC Score
   - Privacy Risk Assessment

### Configuration

Membership Inference Attack can be configured in the `config/security.yaml` file:

```yaml
security:
  # MEMBERSHIP INFERENCE ATTACK (Privacy Attack)
  enable_membership_inference: True
  mia_inference_method: 'confidence_based'    # 'confidence_based', 'threshold_based', 'ml_based'
  mia_test_size: 0.5                          # Proportion of non-members (0.0-1.0)
  mia_threshold_percentile: 50                 # Decision threshold percentile (0-100)
  mia_random_seed: 42                          # Random seed for reproducibility
```

#### Parameter Description

- **`enable_membership_inference`**: Boolean flag to enable/disable MIA
- **`mia_inference_method`**: Inference method (currently 'confidence_based' is used)
- **`mia_test_size`**: Proportion of non-members (0.0-1.0), default: 0.5 (50%)
- **`mia_threshold_percentile`**: Percentile for determining membership score threshold (0-100), default: 50
- **`mia_random_seed`**: Random seed for reproducibility

### Usage

#### Programmatic Usage

```python
from modules.attacks import MembershipInferenceAttack
from modules.rag_network import DRAGNetwork

# Create MIA instance
mia_attack = MembershipInferenceAttack(
    inference_method='confidence_based',
    test_size=0.5,
    threshold_percentile=50,
    random_seed=42
)

# Execute attack
mia_results = mia_attack.execute(network=rag_network, data_points=data_points)

# View results
print(f"Attack Accuracy: {mia_results['attack_accuracy']:.2%}")
print(f"Privacy Risk: {mia_results['privacy_risk']}")
print(f"AUC-ROC: {mia_results['auc_roc']:.4f}")
```

#### Command Line Usage

```bash
# Run simulator with MIA enabled
python simulator.py --security.enable_membership_inference True \
                    --security.mia_test_size 0.3 \
                    --security.mia_threshold_percentile 50
```

### Results and Metrics

After attack execution, the following metrics are available:

```python
{
    "confusion_matrix": {
        "tp": int,  # True Positives
        "tn": int,  # True Negatives
        "fp": int,  # False Positives
        "fn": int   # False Negatives
    },
    "attack_accuracy": float,      # Overall accuracy (0.0-1.0)
    "true_positive_rate": float,    # TPR/Recall (0.0-1.0)
    "false_positive_rate": float,  # FPR (0.0-1.0)
    "precision": float,            # Precision (0.0-1.0)
    "recall": float,               # Recall (0.0-1.0)
    "auc_roc": float,              # AUC-ROC score (0.0-1.0)
    "privacy_risk": str,           # Risk level assessment
    "num_members_tested": int,      # Number of members tested
    "num_non_members_tested": int   # Number of non-members tested
}
```

### Privacy Risk Assessment

Privacy risk level is determined based on the following criteria:

- **CRITICAL**: Risk score > 0.9 - Severe privacy leak
- **HIGH**: Risk score > 0.75 - Significant privacy risk
- **MEDIUM**: Risk score > 0.6 - Moderate privacy concern
- **LOW**: Risk score > 0.5 - Slight privacy risk
- **NEGLIGIBLE**: Risk score ≤ 0.5 - Attack ineffective

Risk score is calculated as follows:
```
risk_score = (accuracy × 0.4) + (TPR × 0.3) + (AUC-ROC × 0.3)
```

### Implementation Details

#### Class Structure

The `MembershipInferenceAttack` class inherits from `BaseAttack` and provides the following methods:

- **`execute()`**: Main attack execution method
- **`_extract_query_features()`**: Query features extraction
- **`_calculate_attack_metrics()`**: Attack metrics calculation
- **`_assess_privacy_risk()`**: Privacy risk assessment
- **`_rebuild_embeddings()`**: Knowledge base embeddings rebuild
- **`evaluate_success()`**: Attack success evaluation

#### Feature Extraction Process

1. A query is sent to the network (`topic_query()`)
2. Answer, num_hops, and num_messages are collected from RAG result
3. Semantic similarity is calculated (`QAEvaluator.calculate_semantic_similarity()`)
4. Answer length ratio and normalized hops are calculated
5. Weighted membership score is created

### Warnings and Limitations

1. **Minimum Data Requirement**: At least 1 member and 1 non-member are required for the attack
2. **Temporary Data Removal**: Non-members are temporarily removed from the knowledge base and embeddings are rebuilt
3. **Network Dependency**: Attack success depends on network topology and retrieval mechanism
4. **Feature Weights**: Feature weights (0.6, 0.3, 0.1) are hardcoded and may require domain-specific tuning

### Example Output

```
Executing MembershipInference attack...
Inference method: confidence_based
Test size (non-members): 0.5
Testing with 50 members and 50 non-members
Extracting features for MEMBERS (data in KB)...
Extracting features for NON-MEMBERS (data NOT in KB)...
Inference threshold (percentile 50): 0.6234
MIA executed: Attack Accuracy = 72.50%, Privacy Risk = HIGH - Significant privacy risk
```

## DEBUG
