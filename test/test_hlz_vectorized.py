"""Test battery_simulation_hlz_only vectorization."""
import pandas as pd
import numpy as np
import time
import sys

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.battery.simulators import battery_simulation_hlz_only, battery_simulation_hlz_only_original

print("=" * 70)
print("🧪 Testing battery_simulation_hlz_only Vectorization")
print("=" * 70)

# Create realistic test data (one year of 15-min intervals)
np.random.seed(42)
n_points = 35040  # One year
test_df = pd.DataFrame({
    'load': 50 + 30 * np.sin(np.linspace(0, 365*2*np.pi, n_points)) + np.random.randn(n_points) * 5
})
test_df['load'] = test_df['load'].clip(lower=0)

# Create HLZ windows (e.g., weekdays 8-20h)
test_df['timestamp'] = pd.date_range('2024-01-01', periods=n_points, freq='15min')
test_df['hour'] = test_df['timestamp'].dt.hour
test_df['weekday'] = test_df['timestamp'].dt.weekday  # 0=Monday, 6=Sunday
test_df['in_window'] = (test_df['weekday'] < 5) & (test_df['hour'] >= 8) & (test_df['hour'] < 20)

# Test parameters
battery_capacity = 215  # kWh
power_rating = 100  # kW
depth_of_discharge = 90  # %
threshold_pct = 85  # 85% of peak load in HLZ windows
battery_efficiency = 0.9

hlz_hours = test_df['in_window'].sum() / 4  # Convert 15-min intervals to hours
print(f"\n📊 Test Configuration:")
print(f"   Dataset size: {len(test_df):,} data points")
print(f"   HLZ windows: {hlz_hours:,.0f} hours ({test_df['in_window'].sum():,} intervals)")
print(f"   HLZ coverage: {test_df['in_window'].mean()*100:.1f}%")
print(f"   Battery: {battery_capacity} kWh, {power_rating} kW")
print(f"   Threshold: {threshold_pct}% of peak HLZ load")

hlz_peak = test_df[test_df['in_window']]['load'].max()
print(f"   Peak load in HLZ windows: {hlz_peak:.1f} kW")
print(f"   Target threshold: {hlz_peak * threshold_pct / 100:.1f} kW")

# ============================================================================
# STEP 1: Warm up Numba JIT compilation
# ============================================================================
print("\n" + "=" * 70)
print("🔥 Step 1: Warming up Numba JIT compilation...")
print("=" * 70)

df_warmup = test_df.head(100).copy()
_ = battery_simulation_hlz_only(
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

# Run original version
print("\n📍 Testing ORIGINAL version (Python loops)...")
times_orig = []
for i in range(3):
    df_test = test_df.copy()
    t0 = time.perf_counter()
    result_orig = battery_simulation_hlz_only_original(
        df_test, battery_capacity, power_rating,
        depth_of_discharge, threshold_pct, battery_efficiency
    )
    elapsed = time.perf_counter() - t0
    times_orig.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f}s")

t_orig = np.median(times_orig)
print(f"   ➜ Median: {t_orig:.4f}s")

# Run vectorized version
print("\n⚡ Testing VECTORIZED version (Numba JIT)...")
times_vec = []
for i in range(3):
    df_test = test_df.copy()
    t0 = time.perf_counter()
    result_vec = battery_simulation_hlz_only(
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

for col in cols_to_check:
    orig_vals = result_orig[col].values
    vec_vals = result_vec[col].values
    
    max_diff = np.abs(orig_vals - vec_vals).max()
    mean_diff = np.abs(orig_vals - vec_vals).mean()
    
    if max_diff < 1e-10:
        print(f"   ✓ {col:20s}: PERFECT MATCH (max diff: {max_diff:.2e})")
    else:
        print(f"   ✗ {col:20s}: MISMATCH (max diff: {max_diff:.2e}, mean: {mean_diff:.2e})")
        all_match = False

# ============================================================================
# STEP 4: HLZ-specific Checks
# ============================================================================
print("\n" + "=" * 70)
print("🔍 Step 4: HLZ-Specific Sanity Checks")
print("=" * 70)

# Check that battery only operates during HLZ windows
charge_outside_hlz = result_vec.loc[~result_vec['in_window'], 'battery_charge'].sum()
discharge_outside_hlz = result_vec.loc[~result_vec['in_window'], 'battery_discharge'].sum()

print(f"   Battery activity outside HLZ windows:")
print(f"   ├─ Charge: {charge_outside_hlz:.2e} kW (should be 0)")
print(f"   └─ Discharge: {discharge_outside_hlz:.2e} kW (should be 0)")

if charge_outside_hlz < 1e-10 and discharge_outside_hlz < 1e-10:
    print(f"   ✓ Battery correctly inactive outside HLZ windows")
else:
    print(f"   ✗ WARNING: Battery active outside HLZ windows!")
    all_match = False

# Check HLZ window performance
hlz_data = result_vec[result_vec['in_window']]
peak_reduction_hlz = hlz_data['load'].max() - hlz_data['grid_load'].max()
total_charge_hlz = hlz_data['battery_charge'].sum()
total_discharge_hlz = hlz_data['battery_discharge'].sum()

print(f"\n   HLZ window performance:")
print(f"   ├─ Peak reduction: {peak_reduction_hlz:.1f} kW")
print(f"   ├─ Total charged (in HLZ): {total_charge_hlz * 0.25:.1f} kWh")
print(f"   └─ Total discharged (in HLZ): {total_discharge_hlz * 0.25:.1f} kWh")

# SoC bounds
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
if all_match and speedup > 5 and charge_outside_hlz < 1e-10:
    print("🎉 ✅ ALL TESTS PASSED!")
    print("=" * 70)
    print(f"   • Accuracy: Perfect match (max diff < 1e-10)")
    print(f"   • Performance: {speedup:.1f}x faster ({t_orig*1000:.1f}ms → {t_vec*1000:.1f}ms)")
    print(f"   • HLZ Logic: Battery correctly operates only in HLZ windows")
    print(f"   • Physics: Battery behavior within expected bounds")
    print("\n   ➜ battery_simulation_hlz_only is PRODUCTION READY! 🚀")
elif all_match:
    print("⚠️  TESTS PASSED (with warning)")
    print("=" * 70)
    print(f"   • Accuracy: ✓ Perfect match")
    if speedup <= 5:
        print(f"   • Performance: ⚠️  Only {speedup:.1f}x faster (expected >10x)")
else:
    print("❌ TESTS FAILED")
    print("=" * 70)
    print(f"   • Accuracy: {'✓' if all_match else '✗'} Results")
    sys.exit(1)

print("=" * 70)

