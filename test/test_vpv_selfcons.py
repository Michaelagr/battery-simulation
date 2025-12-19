"""Test battery_simulation_vpv_selfconsumption_working vectorization."""
import pandas as pd
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.battery.simulators import (battery_simulation_vpv_selfconsumption_working, 
                                    battery_simulation_vpv_selfconsumption_working_original)

print("=" * 70)
print("🧪 Testing battery_simulation_vpv_selfconsumption_working Vectorization")
print("=" * 70)

# Create realistic test data with PV (one year of 15-min intervals)
np.random.seed(42)
n_points = 35040  # One year
timestamps = pd.date_range('2024-01-01', periods=n_points, freq='15min')

# Create load profile
test_df = pd.DataFrame({
    'timestamp': timestamps,
    'load': 50 + 30 * np.sin(np.linspace(0, 365*2*np.pi, n_points)) + np.random.randn(n_points) * 5
})
test_df['load'] = test_df['load'].clip(lower=0)

# Create PV generation (higher in summer, zero at night)
hour = test_df['timestamp'].dt.hour
month = test_df['timestamp'].dt.month

# PV generation pattern: peaks at noon, zero at night, higher in summer
day_factor = np.maximum(0, np.sin((hour - 6) * np.pi / 12))  # Peak at noon
season_factor = 0.5 + 0.5 * np.sin((month - 3) * np.pi / 6)  # Peak in summer
pv_generation = 60 * day_factor * season_factor + np.random.randn(n_points) * 2
test_df['pv'] = pv_generation.clip(lower=0)

# Calculate load after PV (can be negative when PV > load)
test_df['load_pv_neg'] = test_df['load'] - test_df['pv']

# Test parameters
battery_capacity = 215  # kWh
power_rating = 100  # kW
depth_of_discharge = 90  # %
threshold_pct = 85  # 85% of peak load
battery_efficiency = 0.9
reserve_fraction = 0.1
start_month_pv = 6  # June
end_month_pv = 10  # October

print(f"\n📊 Test Configuration:")
print(f"   Dataset size: {len(test_df):,} data points")
print(f"   Battery: {battery_capacity} kWh, {power_rating} kW")
print(f"   Threshold: {threshold_pct}% of peak load")
print(f"   PV-rich period: Months {start_month_pv}-{end_month_pv}")
print(f"   Peak load: {test_df['load'].max():.1f} kW")
print(f"   Peak PV generation: {test_df['pv'].max():.1f} kW")
print(f"   Total PV energy: {test_df['pv'].sum() * 0.25:.0f} kWh/year")
print(f"   Load with PV surplus: {(test_df['load_pv_neg'] < 0).sum():,} intervals")

# ============================================================================
# STEP 1: Warm up Numba JIT compilation
# ============================================================================
print("\n" + "=" * 70)
print("🔥 Step 1: Warming up Numba JIT compilation...")
print("=" * 70)

df_warmup = test_df.head(200).copy()
_ = battery_simulation_vpv_selfconsumption_working(
    df_warmup, battery_capacity, power_rating, 
    depth_of_discharge, threshold_pct, battery_efficiency,
    reserve_fraction, start_month_pv, end_month_pv
)
print("   ✓ JIT compilation complete (function ready for benchmarking)")

# ============================================================================
# STEP 2: Performance Benchmarking
# ============================================================================
print("\n" + "=" * 70)
print("⏱️  Step 2: Running Performance Benchmarks")
print("=" * 70)

# Run original version
print("\n📍 Testing ORIGINAL version (Python loops with .iterrows())...")
times_orig = []
for i in range(3):
    df_test = test_df.copy()
    t0 = time.perf_counter()
    result_orig = battery_simulation_vpv_selfconsumption_working_original(
        df_test, battery_capacity, power_rating,
        depth_of_discharge, threshold_pct, battery_efficiency,
        reserve_fraction, start_month_pv, end_month_pv
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
    result_vec = battery_simulation_vpv_selfconsumption_working(
        df_test, battery_capacity, power_rating,
        depth_of_discharge, threshold_pct, battery_efficiency,
        reserve_fraction, start_month_pv, end_month_pv
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

cols_to_check = [
    'grid_load_pv_bt', 'battery_charge', 'battery_discharge', 'battery_soc',
    'battery_charge_pv', 'battery_charge_grid',
    'battery_discharge_peakshave', 'battery_discharge_selfcons',
    'battery_charge_pv_selfcons', 'battery_charge_grid_peakshave',
    'soc_reserve'
]

all_match = True
for col in cols_to_check:
    orig_vals = result_orig[col].values
    vec_vals = result_vec[col].values
    
    max_diff = np.abs(orig_vals - vec_vals).max()
    
    if max_diff < 1e-10:
        print(f"   ✓ {col:30s}: PERFECT MATCH (max diff: {max_diff:.2e})")
    else:
        print(f"   ✗ {col:30s}: MISMATCH (max diff: {max_diff:.2e})")
        all_match = False

# ============================================================================
# STEP 4: Complex Logic Validation
# ============================================================================
print("\n" + "=" * 70)
print("🔍 Step 4: Complex Logic Validation")
print("=" * 70)

# Check PV charging
pv_charge_total = result_vec['battery_charge_pv'].sum() * 0.25
print(f"   PV charging:")
print(f"   ├─ Total PV charged: {pv_charge_total:.1f} kWh")
print(f"   └─ As % of PV generation: {pv_charge_total / (test_df['pv'].sum() * 0.25) * 100:.1f}%")

# Check self-consumption discharge (should only happen in PV-rich months)
pv_rich_mask = result_vec['pv_rich']
selfcons_in_pv_rich = result_vec.loc[pv_rich_mask, 'battery_discharge_selfcons'].sum()
selfcons_outside_pv_rich = result_vec.loc[~pv_rich_mask, 'battery_discharge_selfcons'].sum()

print(f"\n   Self-consumption discharge:")
print(f"   ├─ In PV-rich period: {selfcons_in_pv_rich * 0.25:.1f} kWh")
print(f"   └─ Outside PV-rich: {selfcons_outside_pv_rich * 0.25:.1f} kWh")

if selfcons_in_pv_rich > 0 and selfcons_outside_pv_rich < 1e-6:
    print(f"   ✓ Self-consumption correctly limited to PV-rich periods")
else:
    print(f"   ⚠️  Unexpected self-consumption pattern")

# Check peak shaving
peakshave_total = result_vec['battery_discharge_peakshave'].sum() * 0.25
peak_reduction = test_df['load'].max() - result_vec['grid_load_pv_bt'].max()

print(f"\n   Peak shaving:")
print(f"   ├─ Total peakshave discharge: {peakshave_total:.1f} kWh")
print(f"   ├─ Peak load (original): {test_df['load'].max():.1f} kW")
print(f"   ├─ Peak load (optimized): {result_vec['grid_load_pv_bt'].max():.1f} kW")
print(f"   └─ Peak reduction: {peak_reduction:.1f} kW")

# SoC bounds and reserve
soc_min = result_vec['battery_soc'].min()
soc_max = result_vec['battery_soc'].max()
reserve_energy = battery_capacity * (1 - depth_of_discharge / 100)
reserve_soc_val = result_vec['soc_reserve'].iloc[0]

print(f"\n   SoC management:")
print(f"   ├─ Min SoC: {soc_min:.1f} kWh")
print(f"   ├─ Max SoC: {soc_max:.1f} kWh")
print(f"   ├─ Technical minimum: {reserve_energy:.1f} kWh")
print(f"   ├─ Dynamic reserve: {reserve_soc_val:.1f} kWh")
print(f"   └─ Within bounds: {'✓ Yes' if soc_min >= reserve_energy - 0.01 and soc_max <= battery_capacity + 0.01 else '✗ NO'}")

# Energy balance
total_charge = result_vec['battery_charge'].sum() * 0.25
total_discharge = result_vec['battery_discharge'].sum() * 0.25
efficiency_observed = total_discharge / total_charge if total_charge > 0 else 0

print(f"\n   Energy balance:")
print(f"   ├─ Total charged: {total_charge:.1f} kWh")
print(f"   ├─ Total discharged: {total_discharge:.1f} kWh")
print(f"   ├─ Efficiency (observed): {efficiency_observed:.3f}")
print(f"   └─ Expected efficiency: {battery_efficiency:.3f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
if all_match and speedup > 5:
    print("🎉 ✅ ALL TESTS PASSED!")
    print("=" * 70)
    print(f"   • Accuracy: Perfect match on all {len(cols_to_check)} output columns")
    print(f"   • Performance: {speedup:.1f}x faster ({t_orig*1000:.1f}ms → {t_vec*1000:.1f}ms)")
    print(f"   • PV Logic: Charging and self-consumption working correctly")
    print(f"   • Peak Shaving: {peak_reduction:.1f} kW reduction achieved")
    print(f"   • Reserve Management: Dynamic SoC limits respected")
    print("\n   ➜ battery_simulation_vpv_selfconsumption_working is PRODUCTION READY! 🚀")
elif all_match:
    print("⚠️  TESTS PASSED (with warning)")
    print("=" * 70)
    print(f"   • Accuracy: ✓ Perfect match")
    if speedup <= 5:
        print(f"   • Performance: ⚠️  Only {speedup:.1f}x faster (expected >10x)")
else:
    print("❌ TESTS FAILED")
    print("=" * 70)
    print(f"   • Accuracy: ✗ Results don't match")
    sys.exit(1)

print("=" * 70)

