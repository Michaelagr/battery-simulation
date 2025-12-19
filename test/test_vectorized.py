"""Quick test to verify vectorized battery simulation matches original."""
import pandas as pd
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.battery.simulators import battery_simulation_ps, battery_simulation_ps_original

# Create realistic test data (one year of 15-min intervals)
np.random.seed(42)  # For reproducibility
n_points = 35040  # One year of 15-minute intervals
test_df = pd.DataFrame({
    'net_load_kw': 50 + 30 * np.sin(np.linspace(0, 365*2*np.pi, n_points)) + np.random.randn(n_points) * 5
})

# Test parameters
battery_capacity = 215  # kWh
power_rating = 100  # kW
threshold_kw = 20  # kW reduction target
depth_of_discharge = 90  # %
battery_efficiency = 0.9

print("🧪 Testing vectorized vs original implementation...")
print(f"   Dataset size: {len(test_df):,} points")

# ⚠️ CRITICAL: Warm up Numba JIT compilation (don't time this!)
print("\n🔥 Warming up Numba JIT compilation...")
df_warmup = test_df.head(100).copy()
_ = battery_simulation_ps(df_warmup, battery_capacity, power_rating, threshold_kw, depth_of_discharge, battery_efficiency)
print("   ✓ JIT compilation complete")

# Now run the REAL benchmark
print("\n⏱️  Running timed benchmarks...")

# Run original version (multiple times for accuracy)
times_original = []
for i in range(3):
    df_original = test_df.copy()
    t0 = time.perf_counter()
    result_original = battery_simulation_ps_original(
        df_original, battery_capacity, power_rating, threshold_kw, 
        depth_of_discharge, battery_efficiency
    )
    times_original.append(time.perf_counter() - t0)

t_original = np.median(times_original)
print(f"   Original: {t_original:.4f}s (median of 3 runs)")

# Run vectorized version (multiple times for accuracy)
times_vectorized = []
for i in range(3):
    df_vectorized = test_df.copy()
    t0 = time.perf_counter()
    result_vectorized = battery_simulation_ps(
        df_vectorized, battery_capacity, power_rating, threshold_kw,
        depth_of_discharge, battery_efficiency
    )
    times_vectorized.append(time.perf_counter() - t0)

t_vectorized = np.median(times_vectorized)
print(f"   Vectorized: {t_vectorized:.4f}s (median of 3 runs)")

speedup = t_original / t_vectorized
if speedup > 1:
    print(f"   ⚡ Speedup: {speedup:.1f}x FASTER!")
else:
    print(f"   ⚠️  Slowdown: {1/speedup:.1f}x SLOWER (unexpected!)")

# Verify results match
print("\n✅ Validating results...")
cols_to_check = ['ps_grid_load', 'battery_charge', 'battery_discharge', 'battery_soc']

all_match = True
for col in cols_to_check:
    try:
        np.testing.assert_allclose(
            result_original[col].values,
            result_vectorized[col].values,
            rtol=1e-10,  # Very tight tolerance
            atol=1e-10
        )
        max_diff = np.abs(result_original[col].values - result_vectorized[col].values).max()
        print(f"   ✓ {col}: MATCH (max diff: {max_diff:.2e})")
    except AssertionError as e:
        all_match = False
        max_diff = np.abs(result_original[col].values - result_vectorized[col].values).max()
        print(f"   ✗ {col}: MISMATCH (max diff: {max_diff:.2e})")

if all_match:
    print("\n🎉 SUCCESS! Vectorized version produces identical results!")
    if speedup > 1:
        print(f"   Performance gain: {speedup:.1f}x faster")
    else:
        print(f"   ⚠️  Performance: {1/speedup:.1f}x slower (may need further optimization)")
else:
    print("\n❌ FAILED! Results don't match. Check implementation.")