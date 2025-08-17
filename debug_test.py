#!/usr/bin/env python3

import sys
import os
import time

print("🔍 Starting debug test...")

# Test 1: Basic imports
print("Test 1: Testing basic imports...")
try:
    import numpy as np
    print("✅ numpy imported")
except Exception as e:
    print(f"❌ numpy import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ pandas imported")
except Exception as e:
    print(f"❌ pandas import failed: {e}")
    sys.exit(1)

# Test 2: CSV reading
print("\nTest 2: Testing CSV reading...")
try:
    csv_path = "Dataset/fertilizers.csv"
    df = pd.read_csv(csv_path, sep=';')
    print(f"✅ CSV read successfully: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except Exception as e:
    print(f"❌ CSV reading failed: {e}")
    sys.exit(1)

# Test 3: Single column extraction
print("\nTest 3: Testing single column extraction...")
try:
    column_name = 'china_k2o'
    if column_name in df.columns:
        series = df[column_name].dropna().values
        print(f"✅ Column extracted: {len(series)} points")
        print(f"Range: {series.min():.2f} to {series.max():.2f}")
    else:
        print(f"❌ Column {column_name} not found")
        sys.exit(1)
except Exception as e:
    print(f"❌ Column extraction failed: {e}")
    sys.exit(1)

# Test 4: Data validation
print("\nTest 4: Testing data validation...")
try:
    if len(series) < 10:
        print(f"❌ Time series too short: {len(series)} points")
        sys.exit(1)
    
    if np.any(np.isnan(series)):
        print("❌ Found NaN values")
        sys.exit(1)
    
    if np.any(np.isinf(series)):
        print("❌ Found infinite values")
        sys.exit(1)
    
    if np.std(series) == 0:
        print("❌ No variation in data")
        sys.exit(1)
    
    print("✅ Data validation passed")
except Exception as e:
    print(f"❌ Data validation failed: {e}")
    sys.exit(1)

# Test 5: Arrow conversion
print("\nTest 5: Testing Arrow conversion...")
try:
    from scripts.generate_ts import convert_to_arrow
    print("✅ Arrow conversion module imported")
    
    time_series = [series.astype(np.float32)]
    arrow_path = "test_debug.arrow"
    
    print("Converting to Arrow...")
    convert_to_arrow(arrow_path, time_series=time_series)
    print(f"✅ Arrow conversion successful: {arrow_path}")
    
except Exception as e:
    print(f"❌ Arrow conversion failed: {e}")
    sys.exit(1)

# Test 6: Arrow loading
print("\nTest 6: Testing Arrow loading...")
try:
    from gluonts.dataset.arrow import ArrowFile
    print("✅ ArrowFile imported")
    
    print("Opening Arrow file...")
    test_dataset = ArrowFile(arrow_path)
    print("✅ ArrowFile created")
    
    print("Converting to list...")
    test_data = list(test_dataset)
    print(f"✅ Data loaded: {len(test_data)} items")
    
    print("Testing batch creation...")
    test_batch = test_data[:1]
    if not test_batch or test_batch[0] is None:
        print("❌ Batch creation failed")
        sys.exit(1)
    print("✅ Batch creation successful")
    
except Exception as e:
    print(f"❌ Arrow loading failed: {e}")
    sys.exit(1)

print("\n🎉 All tests passed! The issue must be elsewhere in the main script.")
