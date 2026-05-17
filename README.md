# Part 1 – Neural Network Analysis: Customer Churn Prediction

## Overview
A feed-forward neural network trained on a synthetic customer churn dataset (2,000 rows, 16 features) to predict whether a customer is likely to leave.

## Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb          ← Full analysis (Tasks 1–6)
├── requirements.txt
└── results/
    ├── class_distribution.png
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

## Setup
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Key Findings
| Experiment | Architecture | LR | Activation | Test Acc | Test F1 |
|---|---|---|---|---|---|
| Exp 1 – Baseline | 64 | 0.001 | ReLU | 0.9850 | 0.0000 |
| **Exp 2 – Deeper** | **128→64** | **0.001** | **ReLU** | **0.9875** | **0.2857** |
| Exp 3 – High LR | 64 | 0.01 | ReLU | 0.9850 | 0.0000 |
| Exp 4 – Low LR | 64 | 0.0001 | ReLU | 0.9850 | 0.0000 |
| Exp 5 – tanh | 64 | 0.001 | tanh | 0.9850 | 0.0000 |

**Best model:** Experiment 2 (deeper network, LR=0.001, ReLU) with F1=0.29.

**Note:** High accuracy across all models is driven by class imbalance (98.45% non-churn). F1-score is the primary metric.

## Dataset
`customer_churn_nn.csv` – 2000 rows, 16 features, binary target (`churn`).

## Source Link: https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing 
Find the relevant datasets for Part 1
