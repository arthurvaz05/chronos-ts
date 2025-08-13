# 📊 Metrics Aggregation Scripts

This directory contains scripts to automatically find and combine all metrics comparison files from all result folders into a single comprehensive CSV file.

## 🚀 Quick Start

### Option 1: Quick Combine (Recommended)
```bash
python scripts/quick_combine_metrics.py
```

### Option 2: Full Featured Combine
```bash
python scripts/combine_all_metrics.py
```

## 📁 What These Scripts Do

1. **Automatically Discover**: Find all `*_metrics_comparison.csv` files in the results directory
2. **Smart Filtering**: Exclude combined/aggregated files to avoid duplicates
3. **Metadata Extraction**: Add Dataset and Column information from file paths
4. **Data Combination**: Merge all metrics into a single comprehensive table
5. **Output Generation**: Save to `results/all_metrics_comparison.csv`

## 🔍 File Discovery Pattern

The scripts look for files matching this pattern:
```
results/
├── dataset_name/
│   ├── column_name/
│   │   └── dataset_column_metrics_comparison.csv
│   └── another_column/
│       └── dataset_another_column_metrics_comparison.csv
└── another_dataset/
    └── column_name/
        └── another_dataset_column_metrics_comparison.csv
```

## 📊 Output Structure

The combined CSV contains:
- **Dataset**: Name of the dataset (e.g., 'climate')
- **Column**: Name of the column (e.g., 'usa_clima')
- **Method**: Prediction method (FT SA, ZR SA, FT RO, ZR RO)
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE (%)**: Mean Absolute Percentage Error
- **MASE**: Mean Absolute Scaled Error
- **SMAPE (%)**: Symmetric Mean Absolute Percentage Error
- **Correlation**: Correlation coefficient
- **Source_File**: Original file name for reference

## 🎯 Use Cases

### 1. **Research Analysis**
- Compare performance across different datasets
- Analyze which methods work best for different data types
- Generate comprehensive performance reports

### 2. **Model Comparison**
- Side-by-side comparison of Fine-tuned vs Zero-shot
- Steps Ahead vs Rolling Origin performance analysis
- Identify best-performing configurations

### 3. **Data Export**
- Import into Excel for further analysis
- Use in R/Python for statistical analysis
- Generate publication-ready tables

## 🔧 Customization

### Change Output Directory
Modify the `output_path` variable in either script:
```python
output_path = "your/custom/path/all_metrics.csv"
```

### Add More Metrics
If you add new metrics to your comparison files, they'll automatically be included in the combined output.

### Filter Specific Datasets
Modify the file discovery pattern to focus on specific datasets:
```python
pattern = "results/your_dataset/**/*_metrics_comparison.csv"
```

## 📈 Example Output

```
Dataset,Column,Method,RMSE,MAE,MAPE (%),MASE,SMAPE (%),Correlation
climate,china_clima,FT Steps Ahead,0.121,0.106,8.6,0.268,8.5,0.401
climate,china_clima,ZR Steps Ahead,0.082,0.064,5.2,0.161,5.2,0.596
climate,usa_clima,FT Rolling Origin,0.126,0.098,8.6,0.249,8.3,-0.428
climate,usa_clima,ZR Rolling Origin,0.254,0.240,20.6,0.606,18.4,0.600
```

## 🚨 Troubleshooting

### No Files Found
- Ensure you have run the evaluation pipeline first
- Check that metrics comparison files exist in the expected locations
- Verify file naming follows the pattern: `*_metrics_comparison.csv`

### Duplicate Entries
- The scripts automatically filter out combined files
- If you still see duplicates, check for files with names containing 'combined', 'all_metrics', or 'aggregated'

### Path Issues
- Ensure you're running the script from the project root directory
- Check that the `results/` directory exists and contains your data

## 🔄 Automation

You can run these scripts:
- **After each evaluation**: To keep metrics up to date
- **Before analysis**: To prepare data for research
- **In CI/CD pipelines**: To generate automated reports
- **Scheduled**: Using cron jobs or similar schedulers

## 📝 Notes

- The scripts automatically handle missing columns gracefully
- All metrics are sorted by Dataset → Column → Method for easy reading
- Source file information is preserved for traceability
- The combined file overwrites any previous version
