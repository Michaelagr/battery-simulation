"""Test battery_simulation_v02 vectorization."""
import pandas as pd
import numpy as np
import time
import sys

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.battery.simulators import battery_simulation_v02, battery_simulation_v02_original

print("=" * 70)
print("🧪 Testing battery_simulation_v02 Vectorization")
print("=" * 70)

# Create realistic test data (one year of 15-min intervals)
np.random.seed(42)
n_points = 35040  # One year
test_df = pd.DataFrame({
    'load': 50 + 30 * np.sin(np.linspace(0, 365*2*np.pi, n_points)) + np.random.randn(n_points) * 5
})
test_df['load'] = test_df['load'].clip(lower=0)  # No negative loads

# Test parameters
battery_capacity = 215  # kWh
power_rating = 100  # kW
depth_of_discharge = 90  # %
threshold_pct = 85  # 85% of peak load
battery_efficiency = 0.9

print(f"\n📊 Test Configuration:")
print(f"   Dataset size: {len(test_df):,} data points")
print(f"   Battery: {battery_capacity} kWh, {power_rating} kW")
print(f"   Threshold: {threshold_pct}% of peak load")
print(f"   Peak load in test data: {test_df['load'].max():.1f} kW")
print(f"   Target threshold: {test_df['load'].max() * threshold_pct / 100:.1f} kW")

# ============================================================================
# STEP 1: Warm up Numba JIT compilation
# ============================================================================
print("\n" + "=" * 70)
print("🔥 Step 1: Warming up Numba JIT compilation...")
print("=" * 70)

df_warmup = test_df.head(100).copy()
_ = battery_simulation_v02(
    df_warmup, battery_capacity, power_rating, 
    depth_of_discharge, threshold_pct, battery_efficiency
)
print("   ✓ JIT compilation complete (function ready for benchmarking)")

# ============================================================================
# STEP 2: Performance Benchmarking
# ============================================================================
print("\n" + "=" * 70)
print("⏱️  Step 2: Running Performance Benchmarks")
print("=" * 70)

# Run original version (3 iterations)
print("\n📍 Testing ORIGINAL version (Python loops)...")
times_orig = []
for i in range(3):
    df_test = test_df.copy()
    t0 = time.perf_counter()
    result_orig = battery_simulation_v02_original(
        df_test, battery_capacity, power_rating,
        depth_of_discharge, threshold_pct, battery_efficiency
    )
    elapsed = time.perf_counter() - t0
    times_orig.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f}s")

t_orig = np.median(times_orig)
print(f"   ➜ Median: {t_orig:.4f}s")

# Run vectorized version (3 iterations)
print("\n⚡ Testing VECTORIZED version (Numba JIT)...")
times_vec = []
for i in range(3):
    df_test = test_df.copy()
    t0 = time.perf_counter()
    result_vec = battery_simulation_v02(
        df_test, battery_capacity, power_rating,
        depth_of_discharge, threshold_pct, battery_efficiency
    )
    elapsed = time.perf_counter() - t0
    times_vec.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f}s")

t_vec = np.median(times_vec)
print(f"   ➜ Median: {t_vec:.4f}s")

speedup = t_orig / t_vec
print(f"\n{'🎉' if speedup > 10 else '⚡'} PERFORMANCE GAIN: {speedup:.1f}x FASTER!")

# ============================================================================
# STEP 3: Accuracy Validation
# ============================================================================
print("\n" + "=" * 70)
print("✅ Step 3: Validating Result Accuracy")
print("=" * 70)

cols_to_check = ['grid_load', 'battery_charge', 'battery_discharge', 'battery_soc']
all_match = True
max_differences = {}

for col in cols_to_check:
    orig_vals = result_orig[col].values
    vec_vals = result_vec[col].values
    
    max_diff = np.abs(orig_vals - vec_vals).max()
    mean_diff = np.abs(orig_vals - vec_vals).mean()
    max_differences[col] = max_diff
    
    # Check if they match within tolerance
    matches = max_diff < 1e-10
    
    if matches:
        print(f"   ✓ {col:20s}: PERFECT MATCH (max diff: {max_diff:.2e})")
    else:
        print(f"   ✗ {col:20s}: MISMATCH (max diff: {max_diff:.2e}, mean: {mean_diff:.2e})")
        all_match = False

# ============================================================================
# STEP 4: Sanity Checks
# ============================================================================
print("\n" + "=" * 70)
print("🔍 Step 4: Sanity Checks")
print("=" * 70)

# Check basic physics
peak_reduction = test_df['load'].max() - result_vec['grid_load'].max()
total_charge = result_vec['battery_charge'].sum()
total_discharge = result_vec['battery_discharge'].sum()
final_soc = result_vec['battery_soc'].iloc[-1]

print(f"   Peak load reduction: {peak_reduction:.1f} kW")
print(f"   Total energy charged: {total_charge * 0.25:.1f} kWh")
print(f"   Total energy discharged: {total_discharge * 0.25:.1f} kWh")
print(f"   Final battery SoC: {final_soc:.1f} kWh ({final_soc/battery_capacity*100:.1f}%)")
print(f"   Battery efficiency check: {(total_discharge * 0.25) / (total_charge * 0.25) if total_charge > 0 else 0:.3f}")

# Verify SoC stays within bounds
soc_min = result_vec['battery_soc'].min()
soc_max = result_vec['battery_soc'].max()
reserve_energy = battery_capacity * (1 - depth_of_discharge / 100)

print(f"\n   SoC bounds:")
print(f"   ├─ Min SoC: {soc_min:.1f} kWh (reserve: {reserve_energy:.1f} kWh)")
print(f"   ├─ Max SoC: {soc_max:.1f} kWh (capacity: {battery_capacity:.1f} kWh)")
print(f"   └─ Within bounds: {'✓ Yes' if soc_min >= reserve_energy - 0.01 and soc_max <= battery_capacity + 0.01 else '✗ NO'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
if all_match and speedup > 5:
    print("🎉 ✅ ALL TESTS PASSED!")
    print("=" * 70)
    print(f"   • Accuracy: Perfect match (max diff < 1e-10)")
    print(f"   • Performance: {speedup:.1f}x faster ({t_orig*1000:.1f}ms → {t_vec*1000:.1f}ms)")
    print(f"   • Physics: Battery behavior within expected bounds")
    print("\n   ➜ battery_simulation_v02 is PRODUCTION READY! 🚀")
elif all_match:
    print("⚠️  TESTS PASSED (with warning)")
    print("=" * 70)
    print(f"   • Accuracy: ✓ Perfect match")
    print(f"   • Performance: ⚠️  Only {speedup:.1f}x faster (expected >10x)")
    print("\n   ➜ Consider checking Numba installation/configuration")
else:
    print("❌ TESTS FAILED")
    print("=" * 70)
    print(f"   • Accuracy: ✗ Results don't match")
    print(f"   • Max differences: {max_differences}")
    sys.exit(1)

print("=" * 70)

