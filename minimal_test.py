#!/usr/bin/env python3

import sys
import os

print("🔍 Starting minimal test...")

# Test 1: Basic imports
print("Test 1: Testing imports...")
try:
    import yaml
    print("✅ yaml imported")
except Exception as e:
    print(f"❌ yaml import failed: {e}")
    sys.exit(1)

# Test 2: YAML config generation
print("\nTest 2: Testing YAML config generation...")
try:
    dataset_name = "fertilizers"
    column_name = "germany_n"
    prediction_length = 5
    
    config = [
        {
            'name': dataset_name,
            'dataset_path': f'Dataset/Dataset.arrow/{dataset_name}_{column_name}.arrow',
            'offset': -prediction_length,
            'prediction_length': prediction_length,
            'num_rolls': 1
        }
    ]
    config_path = f'scripts/evaluation/configs/{dataset_name}.yaml'
    
    print(f"Creating directory: {os.path.dirname(config_path)}")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    print(f"Writing YAML to: {config_path}")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print("✅ YAML config generated successfully")
    
except Exception as e:
    print(f"❌ YAML config generation failed: {e}")
    sys.exit(1)

# Test 3: Data preparation call
print("\nTest 3: Testing data preparation call...")
try:
    print("Calling prepare_data function...")
    
    # Import the function
    from run_all import prepare_data
    
    csv_path = "Dataset/fertilizers.csv"
    arrow_path = f"Dataset/Dataset.arrow/fertilizers_{column_name}.arrow"
    
    print(f"CSV path: {csv_path}")
    print(f"Arrow path: {arrow_path}")
    
    # Call the function
    result = prepare_data(csv_path, arrow_path, test_mode=False, max_columns=2, column_name=column_name)
    
    print(f"✅ prepare_data returned: {result}")
    
except Exception as e:
    print(f"❌ Data preparation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All tests passed!")
