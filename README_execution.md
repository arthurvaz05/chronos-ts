# Chronos Forecasting - Execution & Evaluation Guide

This guide provides comprehensive instructions for running the main execution files and evaluation scripts in the terminal.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [🚀 MAIN EXECUTION FILES - How to Use in Terminal](#-main-execution-files---how-to-use-in-terminal)
4. [📊 EVALUATION SCRIPTS - Step by Step Usage](#-evaluation-scripts---step-by-step-usage)
5. [🔄 UTILITY SCRIPTS - Data Processing and Visualization](#-utility-scripts---data-processing-and-visualization)
6. [Troubleshooting](#troubleshooting)

## 🚀 Prerequisites

Before running evaluations, ensure you have:

- Python 3.8+ installed
- CUDA-compatible GPU (recommended for faster inference)
- Access to trained Chronos models or pre-trained models from HuggingFace
- Time series datasets in GluonTS Arrow format

## 📦 Installation

### For Standard Evaluation
```bash
pip install "chronos-forecasting[evaluation] @ git+https://github.com/amazon-science/chronos-forecasting.git"
```

### For Training and Local Evaluation
```bash
# Clone the repository
git clone https://github.com/amazon-science/chronos-forecasting.git
cd chronos-forecasting

# Install in editable mode with all dependencies
pip install --editable ".[training,evaluation]"
```

---

## 🚀 MAIN EXECUTION FILES - How to Use in Terminal

### 🎯 **What This Section Covers**
This section explains **exactly how to use the main execution files** that exist in your project:
- `run_all.py` - Master script to run everything
- `evaluate_local_with_predictions.py` - Main evaluation script
- `combine_all_metrics.py` - Combine and aggregate results
- `create_mape_smape_charts.py` - Create visualization charts

### **📁 File Locations**
All these files are located in the `scripts/` directory:
```
scripts/
├── run_all.py                                    # Master execution script
├── evaluation/
│   └── evaluate_local_with_predictions.py        # Main evaluation script
├── combine_all_metrics.py                        # Metrics combination
└── create_mape_smape_charts.py                  # Chart creation
```

---

## 📊 EVALUATION SCRIPTS - Step by Step Usage

### **1. 🎯 `evaluate_local_with_predictions.py` - Main Evaluation Script**

#### **What It Does**
- **Evaluates fine-tuned models** against zero-shot baselines
- **Makes predictions** using different strategies (Steps Ahead, Rolling Origin)
- **Compares performance** between your trained model and base Chronos
- **Generates CSV files** with all predictions and actual values

#### **🚀 Quick Start - Basic Usage**
```bash
# Basic command structure
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "PATH_TO_YOUR_MODEL" \
    --dataset-path "PATH_TO_YOUR_DATASET" \
    --output-path "WHERE_TO_SAVE_RESULTS" \
    --prediction-length 5
```

#### **📋 All Available Parameters**
| Parameter | What It Does | Default | Required | Example |
|-----------|--------------|---------|----------|---------|
| `--model-path` | Path to trained model checkpoint | - | ✅ | `"results/climate/usa_clima/checkpoint-final"` |
| `--dataset-path` | Path to dataset (.arrow file) | - | ✅ | `"Dataset/climate_usa_clima.arrow"` |
| `--output-path` | Where to save results CSV | - | ✅ | `"evaluation/results/my_results.csv"` |
| `--prediction-length` | Number of future values to predict | 5 | ❌ | `--prediction-length 10` |
| `--use-inference-model` | Use larger model for better quality | True | ❌ | `--no-use-inference-model` |
| `--enable-zero-shot` | Compare with zero-shot baseline | True | ❌ | `--no-enable-zero-shot` |

#### **🎯 Real Examples - Climate Models**
```bash
# Example 1: Climate USA Model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# Example 2: Climate China Model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_china_clima.arrow" \
    --output-path "evaluation/results/climate_china_clima_predictions.csv" \
    --prediction-length 5
```

#### **⚡ Performance Optimization Options**
```bash
# Faster execution (no zero-shot comparison)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-enable-zero-shot

# Use trained model directly (saves memory)
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5 \
    --no-use-inference-model

# Custom prediction length
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions_10steps.csv" \
    --prediction-length 10
```

#### **🔄 Batch Processing Multiple Models**
```bash
# Simple loop for multiple models
for model in "usa_clima" "china_clima"; do
    echo "Evaluating $model model..."
    python scripts/evaluation/evaluate_local_with_predictions.py \
        --model-path "results/climate/$model/climate_${model}_training_outputs/run-0/checkpoint-final" \
        --dataset-path "Dataset/climate_${model}.arrow" \
        --output-path "evaluation/results/climate_${model}_predictions.csv" \
        --prediction-length 5
done
```

#### **📊 Understanding the Output**
The script generates a CSV file with this structure:

| timestamp | value | type |
|-----------|-------|------|
| 0 | 1.23 | actual |
| 1 | 1.45 | actual |
| ... | ... | ... |
| 95 | 2.34 | actual |
| 96 | 2.56 | predicted FT SA |
| 97 | 2.78 | predicted FT SA |
| 98 | 2.91 | predicted FT SA |
| 99 | 3.12 | predicted FT SA |
| 100 | 3.34 | predicted FT SA |

**Data Types Explained:**
- **`actual`**: Training and validation data (ground truth)
- **`predicted FT SA`**: Fine-tuned Steps Ahead predictions
- **`predicted ZR SA`**: Zero-shot Steps Ahead predictions  
- **`predicted FT RO`**: Fine-tuned Rolling Origin predictions
- **`predicted ZR RO`**: Zero-shot Rolling Origin predictions

---

## 🔄 UTILITY SCRIPTS - Data Processing and Visualization

### **2. 📊 `combine_all_metrics.py` - Combine All Results**

#### **What It Does**
- **Combines multiple evaluation results** into a single file
- **Aggregates metrics** from different models and datasets
- **Creates summary tables** for easy comparison
- **Exports combined results** in various formats

#### **🚀 Basic Usage**
```bash
# Run the script to combine all available metrics
python scripts/combine_all_metrics.py
```

#### **📋 What It Processes**
The script automatically looks for and combines:
- Results from `evaluate_local_with_predictions.py`
- Standard evaluation results
- Training metrics and parameters
- Performance comparisons

#### **📁 Output Files**
- `evaluation/results/combined_metrics.csv` - All metrics combined
- `evaluation/results/summary_report.txt` - Summary of results
- `evaluation/results/model_comparison.csv` - Model performance comparison

#### **⚙️ Customization Options**
```bash
# Specify custom input/output directories
python scripts/combine_all_metrics.py \
    --input-dir "evaluation/results" \
    --output-dir "evaluation/combined" \
    --format "csv"

# Filter specific metrics
python scripts/combine_all_metrics.py \
    --metrics "mape,smape,rmse" \
    --models "climate,emissions,gdp"
```

### **3. 📈 `create_mape_smape_charts.py` - Create Visualization Charts**

#### **What It Does**
- **Creates MAPE and SMAPE charts** from evaluation results
- **Generates comparison plots** between different models
- **Exports charts** in various formats (PNG, SVG, PDF)
- **Creates interactive visualizations** for analysis

#### **🚀 Basic Usage**
```bash
# Create charts from default results
python scripts/create_mape_smape_charts.py
```

#### **📋 Chart Types Generated**
- **MAPE Comparison Charts**: Mean Absolute Percentage Error across models
- **SMAPE Comparison Charts**: Symmetric Mean Absolute Percentage Error
- **Performance Trend Charts**: How metrics change over time
- **Model Comparison Charts**: Side-by-side performance comparison

#### **⚙️ Customization Options**
```bash
# Specify custom input data
python scripts/create_mape_smape_charts.py \
    --input-file "evaluation/results/combined_metrics.csv" \
    --output-dir "figures/charts"

# Choose chart formats
python scripts/create_mape_smape_charts.py \
    --formats "png,svg,pdf" \
    --dpi 300

# Filter specific models or metrics
python scripts/create_mape_smape_charts.py \
    --models "climate_usa,climate_china" \
    --metrics "mape,smape"
```

#### **📁 Output Files**
- `figures/mape_comparison.png` - MAPE comparison chart
- `figures/smape_comparison.png` - SMAPE comparison chart
- `figures/model_performance.png` - Overall model performance
- `figures/interactive_charts.html` - Interactive web charts

---

## 🚀 COMPLETE WORKFLOW - Run Everything in Sequence

### **Step 1: Run Individual Evaluations**
```bash
# Evaluate Climate USA model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_usa_clima.arrow" \
    --output-path "evaluation/results/climate_usa_clima_predictions.csv" \
    --prediction-length 5

# Evaluate Climate China model
python scripts/evaluation/evaluate_local_with_predictions.py \
    --model-path "results/climate/china_clima/climate_china_clima_training_outputs/run-0/checkpoint-final" \
    --dataset-path "Dataset/climate_china_clima.arrow" \
    --output-path "evaluation/results/climate_china_clima_predictions.csv" \
    --prediction-length 5
```

### **Step 2: Combine All Metrics**
```bash
# Combine all evaluation results
python scripts/combine_all_metrics.py
```

### **Step 3: Create Visualization Charts**
```bash
# Generate MAPE and SMAPE charts
python scripts/create_mape_smape_charts.py
```

### **🔄 One-Command Execution (if run_all.py exists)**
```bash
# If you have a run_all.py script, you can run everything at once
python scripts/run_all.py
```

---

## 🛠️ Troubleshooting

### **Common Issues and Solutions**

#### **1. Model Loading Errors**
```bash
# Error: "Failed to load model"
# Solution: Check if model path exists
ls -la results/climate/usa_clima/climate_usa_clima_training_outputs/run-0/checkpoint-final/
```

#### **2. Dataset Format Issues**
```bash
# Error: "Failed to load dataset"
# Solution: Ensure dataset is in GluonTS Arrow format
ls -la Dataset/climate_usa_clima.arrow
```

#### **3. Memory Issues**
```bash
# Solution: Use smaller models or reduce batch size
--no-use-inference-model
--no-enable-zero-shot
```

#### **4. Missing Dependencies**
```bash
# Install missing packages
pip install torch torchvision torchaudio
pip install gluonts
pip install datasets
pip install scipy
pip install matplotlib
pip install pandas
```

### **Performance Optimization**

#### **GPU Usage**
```bash
# Check GPU availability
nvidia-smi

# Force GPU usage
export CUDA_VISIBLE_DEVICES=0
```

#### **Memory Management**
```bash
# Use trained model directly (saves memory)
--no-use-inference-model

# Disable zero-shot for faster execution
--no-enable-zero-shot

# Reduce prediction length
--prediction-length 3
```

---

## 📚 Additional Resources

- [Main Chronos Documentation](../README.md)
- [Training Guide](../scripts/README.md)
- [Model Architecture Details](../docs/)
- [Paper: Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)

## 🤝 Support

If you encounter issues:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [GitHub Issues](https://github.com/amazon-science/chronos-forecasting/issues)
3. Create a new issue with detailed error information
4. Include your system configuration and error logs

---

**Happy Forecasting! 🚀**
