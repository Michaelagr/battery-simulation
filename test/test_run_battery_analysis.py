"""Test run_battery_analysis vectorization - CRITICAL validation test."""
import pandas as pd
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loaders import read_load_profile, read_price_data, load_solar_data
from src.battery.simulators import battery_simulation_ps
from src.battery.analysis import _smart_battery_core
from src.config import (INTERVAL_HOURS, BATTERY_EFFICIENCY, DEFAULT_DEPTH_OF_DISCHARGE,
                        MIN_ARBITRAGE_SPREAD, NEGATIVE_PRICE_THRESHOLD, ENABLE_ARBITRAGE)

print("=" * 80)
print("🧪 CRITICAL TEST: run_battery_analysis() Vectorization Validation")
print("=" * 80)
print("\n⚠️  This test validates that the vectorized main simulation loop")
print("   produces IDENTICAL results to the original 196-line Python loop.")
print("=" * 80)

# Use actual data files for realistic testing
test_file = "input/demo/01_Demo_small_2024.xlsx"

print(f"\n📊 Test Configuration:")
print(f"   Input file: {test_file}")
print(f"   Testing with real data for maximum confidence")

# Load and prepare data (same as run_battery_analysis)
print("\n📥 Loading data...")
df_load = read_load_profile(test_file)
df_prices = read_price_data(2024)

df_load['merge_key'] = df_load['timestamp'].dt.strftime('%m-%d %H:%M:%S')
df_prices['merge_key'] = df_prices['timestamp'].dt.strftime('%m-%d %H:%M:%S')

pv_capacity = 100  # Test with PV
df_pv = load_solar_data(pv_capacity, None)
df_pv['merge_key'] = df_pv['timestamp'].dt.strftime('%m-%d %H:%M:%S')

df_merged = pd.merge(df_load, df_prices[['merge_key', 'price']], on='merge_key', how='left')
df_merged = pd.merge(df_merged, df_pv[['merge_key', 'yearly_production_kw', 'yearly_production_kwh']], on='merge_key', how='left')
df_merged['load_pv'] = df_merged['load_org'] - df_merged['yearly_production_kw']
df_merged['net_load_kw'] = df_merged['load_pv']
df_merged['net_load_kwh'] = df_merged['load_pv'] * INTERVAL_HOURS

# Battery parameters
battery_power_kw = 100
battery_capacity_kwh = 215
peak_shaving_capacity_percent = 100
interval_hours = INTERVAL_HOURS
depth_of_discharge = 0.1 * battery_capacity_kwh
arbitrage_enabled = ENABLE_ARBITRAGE

print(f"   Dataset size: {len(df_merged):,} intervals")
print(f"   Battery: {battery_capacity_kwh} kWh, {battery_power_kw} kW")
print(f"   Peak shaving capacity: {peak_shaving_capacity_percent}%")
print(f"   Arbitrage: {'Enabled' if arbitrage_enabled else 'Disabled'}")

# Prepare data
df_case_4 = df_merged.copy()
df_case_4 = df_case_4.reset_index(drop=True)

# Find optimal peak shaving threshold
peak_load = df_case_4["net_load_kw"].max()
total_energy_kwh_case_4 = df_case_4["net_load_kwh"].sum()

print(f"\n🔍 Finding optimal peak shaving threshold...")
all_results = []
min_reduction_ps = min(0.2 * battery_power_kw, 10)
max_reduction_ps = min(battery_power_kw, peak_load)
reduction_values_ps = np.arange(min_reduction_ps, max_reduction_ps, 1)

best_peak_reduction = -np.inf

for ps_reduction_value in reduction_values_ps:
    df_test = df_case_4.copy()
    df_ps = battery_simulation_ps(
        df_test, battery_capacity_kwh, battery_power_kw, 
        threshold_kw=ps_reduction_value, depth_of_discharge=DEFAULT_DEPTH_OF_DISCHARGE, 
        battery_efficiency=BATTERY_EFFICIENCY
    )
    peak_after_ps = df_ps["ps_grid_load"].max()
    peak_reduction_ps = peak_load - peak_after_ps
    threshold_kw = peak_load - peak_reduction_ps
    
    all_results.append({
        "threshold_kW": threshold_kw,
        "peak_after_ps": peak_after_ps,
        "kW_reduction": peak_reduction_ps
    })
    
    if peak_reduction_ps >= best_peak_reduction:
        best_peak_reduction = peak_reduction_ps
        best_threshold_ps_only = threshold_kw
    else:
        break

best = max(all_results, key=lambda x: x["kW_reduction"])
best_threshold_ps_only = best["threshold_kW"]
peak_shaving_threshold_kw = best_threshold_ps_only

print(f"   ✓ Optimal threshold: {peak_shaving_threshold_kw:.1f} kW")

# Prepare look-ahead and other pre-computed values
min_soc_kwh = depth_of_discharge if peak_shaving_capacity_percent > 3 else depth_of_discharge + battery_capacity_kwh * 0.1
other_uses_max_soc_kwh = battery_capacity_kwh * (1 - peak_shaving_capacity_percent / 100)

look_ahead_intervals_peak = 96 if battery_capacity_kwh < 431 else 96*2
look_ahead_intervals_price = 96
low_price_perc = 30
high_price_perc = 70

# Precompute look-ahead values
excess_kw = np.clip(df_case_4['net_load_kw'] - peak_shaving_threshold_kw, 0, None)
df_case_4['future_excess_power_kw'] = (
    pd.Series(excess_kw).rolling(look_ahead_intervals_peak, min_periods=1).max().shift(-look_ahead_intervals_peak + 1).to_numpy())
df_case_4['future_excess_energy_kwh'] = (
    pd.Series(excess_kw).rolling(look_ahead_intervals_peak, min_periods=1).sum().shift(-look_ahead_intervals_peak + 1).to_numpy() * interval_hours)

s = df_case_4['net_load_kw'].to_numpy()
fw_max = (pd.Series(s[::-1]).rolling(look_ahead_intervals_peak, min_periods=1).max()[::-1]).to_numpy()
df_case_4['future_peak_max_kw'] = fw_max

from numpy.lib.stride_tricks import sliding_window_view
def fast_forward_quantile(arr, window, q):
    padded = np.pad(arr, (0, window-1), constant_values=np.nan)
    windows = sliding_window_view(padded, window)
    return np.nanpercentile(windows, q, axis=1)

prices = df_case_4['price'].to_numpy()
prices_for_high = np.where(prices > 0, prices, 0.01)
df_case_4['future_price_low'] = fast_forward_quantile(prices, look_ahead_intervals_price, low_price_perc)
df_case_4['future_price_high'] = fast_forward_quantile(prices_for_high, look_ahead_intervals_price, high_price_perc)

df_case_4 = df_case_4.reset_index(drop=True)

# Prepare numpy arrays
net_load_kwh = df_case_4['net_load_kwh'].to_numpy()
net_load_kw_np = df_case_4['net_load_kw'].to_numpy()
price = df_case_4['price'].to_numpy()
future_price_low = df_case_4['future_price_low'].to_numpy()
future_price_high = df_case_4['future_price_high'].to_numpy()
future_peak_max_kw = df_case_4['future_peak_max_kw'].to_numpy()
future_excess_energy_kwh = df_case_4['future_excess_energy_kwh'].to_numpy()

print(f"\n" + "=" * 80)
print(f"🔥 Warming up Numba JIT compilation...")
print("=" * 80)

# Warm up with small dataset
_ = _smart_battery_core(
    net_load_kw_np[:100], net_load_kwh[:100], price[:100], 
    future_price_low[:100], future_price_high[:100],
    future_peak_max_kw[:100], future_excess_energy_kwh[:100],
    peak_shaving_threshold_kw, battery_capacity_kwh, battery_power_kw,
    min_soc_kwh, other_uses_max_soc_kwh, interval_hours,
    peak_shaving_capacity_percent, peak_load, arbitrage_enabled,
    BATTERY_EFFICIENCY
)
print("   ✓ JIT compilation complete")

# ============================================================================
# RUN ORIGINAL VERSION (Python loop)
# ============================================================================
print("\n" + "=" * 80)
print("📍 Step 1: Running ORIGINAL version (196-line Python loop)")
print("=" * 80)

battery_soc_kwh_orig = battery_capacity_kwh
battery_power_kwh = battery_power_kw * interval_hours

battery_discharge_kw_orig = np.zeros(len(df_case_4))
battery_discharge_ps_kw_orig = np.zeros(len(df_case_4))
battery_discharge_ls_kw_orig = np.zeros(len(df_case_4))
battery_soc_kwh_arr_orig = np.zeros(len(df_case_4))
grid_import_kw_orig = np.zeros(len(df_case_4))
grid_export_kw_orig = np.zeros(len(df_case_4))
grid_export_pv_kw_orig = np.zeros(len(df_case_4))
grid_import_avoided_arbitrage_kw_orig = np.zeros(len(df_case_4))
battery_charge_kw_orig = np.zeros(len(df_case_4))
arbitrage_charge_energy_kwh_orig = np.zeros(len(df_case_4))
arbitrage_discharge_energy_kwh_orig = np.zeros(len(df_case_4))
arbitrage_charge_prices_orig = np.zeros(len(df_case_4))
arbitrage_discharge_prices_orig = np.zeros(len(df_case_4))

t0 = time.perf_counter()

for idx in range(len(df_case_4)):
    net_load_kw = net_load_kw_np[idx]
    net_load_kwh_interval = net_load_kwh[idx]
    current_price = price[idx]
    low_price_threshold = future_price_low[idx]
    high_price_threshold = future_price_high[idx]
    future_peak_value = future_peak_max_kw[idx]
    required_energy_for_peak = future_excess_energy_kwh[idx]
    
    battery_discharge_kwh_interval = 0.0
    battery_discharge_kwh_interval_ls = 0.0
    battery_discharge_kwh_interval_ps = 0.0
    grid_import_kwh_interval = 0.0
    grid_export_kwh_interval = 0.0
    grid_export_pv_kwh_interval = 0.0
    grid_import_avoided_arbitrage_kwh_interval = 0.0
    battery_charge_kwh_interval = 0.0
    battery_charge_kwh_interval_pv = 0.0
    battery_charge_kwh_interval_grid = 0.0
    
    grid_import_kwh_interval = max(0, net_load_kwh_interval)
    action_taken = 0
    
    # 1. Peak Shaving
    if net_load_kw > peak_shaving_threshold_kw:
        discharge_needed_kwh = (net_load_kw - peak_shaving_threshold_kw) * interval_hours
        peak_shaving_min_limit = max(depth_of_discharge, other_uses_max_soc_kwh)
        max_discharge_possible_kwh = min(battery_power_kwh, (battery_soc_kwh_orig - peak_shaving_min_limit))
        actual_discharge_kwh = min(discharge_needed_kwh, max(0, max_discharge_possible_kwh))
        
        if actual_discharge_kwh > 0:
            battery_discharge_kwh_interval = actual_discharge_kwh
            battery_discharge_kwh_interval_ps += actual_discharge_kwh
            battery_soc_kwh_orig -= battery_discharge_kwh_interval
            grid_import_kwh_interval = max(0, net_load_kwh_interval - battery_discharge_kwh_interval)
            action_taken = 1
    
    # 2. PV Surplus Charging
    elif net_load_kw < 0 and action_taken == 0:
        charge_potential_kwh_interval = abs(net_load_kwh_interval)
        max_charge_possible_kwh_interval = min(battery_power_kwh, battery_capacity_kwh - battery_soc_kwh_orig)
        actual_charge_from_PV_kwh_interval = min(charge_potential_kwh_interval, max(0, max_charge_possible_kwh_interval))
        
        if actual_charge_from_PV_kwh_interval > 0:
            battery_charge_kwh_interval_pv = actual_charge_from_PV_kwh_interval
            battery_soc_kwh_orig += battery_charge_kwh_interval_pv * BATTERY_EFFICIENCY
            grid_export_kwh_interval = charge_potential_kwh_interval - battery_charge_kwh_interval_pv
            grid_export_pv_kwh_interval = grid_export_kwh_interval
            grid_import_kwh_interval = 0
            action_taken = 1
    
    # 3. Arbitrage and Emergency Charging
    else:
        future_peak_detected = future_peak_value > peak_shaving_threshold_kw
        peak_reserve_soc = min(battery_capacity_kwh, min_soc_kwh + required_energy_for_peak) if peak_shaving_capacity_percent > 0 else min_soc_kwh
        
        if peak_shaving_capacity_percent > 0 and future_peak_detected and battery_soc_kwh_orig < peak_reserve_soc:
            max_charge_without_exceeding_threshold = max(0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
            charge_amount_kwh = min(
                max_charge_without_exceeding_threshold,
                battery_power_kw * interval_hours,
                battery_capacity_kwh - battery_soc_kwh_orig,
                peak_reserve_soc - battery_soc_kwh_orig)
            
            if charge_amount_kwh > 0:
                battery_charge_kwh_interval_grid = charge_amount_kwh
                battery_soc_kwh_orig += charge_amount_kwh * BATTERY_EFFICIENCY
                grid_import_kwh_interval += charge_amount_kwh
                action_taken = 1
        
        min_profitable_discharge_price = current_price / BATTERY_EFFICIENCY + MIN_ARBITRAGE_SPREAD
        arbitrage_is_profitable = high_price_threshold >= min_profitable_discharge_price
        is_very_negative_price = current_price <= NEGATIVE_PRICE_THRESHOLD
        should_charge = (current_price <= low_price_threshold and arbitrage_is_profitable) or is_very_negative_price
        
        if should_charge and battery_soc_kwh_orig < battery_capacity_kwh and arbitrage_enabled:
            if peak_shaving_capacity_percent > 0:
                max_charge_without_exceeding_threshold = max(
                    0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
            else:
                max_charge_without_exceeding_threshold = max(
                    0, (peak_load - grid_import_kwh_interval / interval_hours) * interval_hours)
            
            charge_amount_kwh = min(
                max_charge_without_exceeding_threshold,
                battery_power_kwh,
                battery_capacity_kwh - battery_soc_kwh_orig
            )
            if charge_amount_kwh > 0:
                battery_charge_kwh_interval_grid += charge_amount_kwh
                battery_soc_kwh_orig += charge_amount_kwh * BATTERY_EFFICIENCY
                grid_import_kwh_interval += charge_amount_kwh
                arbitrage_charge_energy_kwh_orig[idx] = charge_amount_kwh
                arbitrage_charge_prices_orig[idx] = current_price
                action_taken = 1
        
        if low_price_threshold < 0:
            discharge_is_profitable = current_price > 0
        else:
            min_profitable_current_price = low_price_threshold / BATTERY_EFFICIENCY + MIN_ARBITRAGE_SPREAD
            discharge_is_profitable = current_price >= min_profitable_current_price
        
        min_reserve_for_check = min_soc_kwh if peak_shaving_capacity_percent == 0 else peak_reserve_soc
        if (current_price >= high_price_threshold and action_taken == 0 and
            battery_soc_kwh_orig > min_reserve_for_check and 
            (battery_charge_kwh_interval_pv + battery_charge_kwh_interval_grid) == 0.0 and
            grid_import_kwh_interval > 0 and discharge_is_profitable and arbitrage_enabled):
            
            required_reserve = min_soc_kwh if peak_shaving_capacity_percent == 0 else max(min_soc_kwh, peak_reserve_soc)
            max_discharge_for_arbitrage = max(0.0, battery_soc_kwh_orig - required_reserve)
            discharge_amount_kwh = min(battery_power_kwh, max_discharge_for_arbitrage, grid_import_kwh_interval)
            
            if discharge_amount_kwh > 0:
                battery_soc_kwh_orig -= discharge_amount_kwh
                battery_discharge_kwh_interval += discharge_amount_kwh
                battery_discharge_kwh_interval_ls += discharge_amount_kwh
                grid_import_kwh_interval -= discharge_amount_kwh
                grid_import_avoided_arbitrage_kwh_interval = discharge_amount_kwh
                arbitrage_discharge_energy_kwh_orig[idx] = discharge_amount_kwh
                arbitrage_discharge_prices_orig[idx] = current_price
    
    battery_discharge_kw_orig[idx] = battery_discharge_kwh_interval / interval_hours
    battery_discharge_ls_kw_orig[idx] = battery_discharge_kwh_interval_ls / interval_hours
    battery_discharge_ps_kw_orig[idx] = battery_discharge_kwh_interval_ps / interval_hours
    battery_soc_kwh_arr_orig[idx] = battery_soc_kwh_orig
    grid_import_kw_orig[idx] = grid_import_kwh_interval / interval_hours
    grid_export_kw_orig[idx] = grid_export_kwh_interval / interval_hours
    grid_export_pv_kw_orig[idx] = grid_export_pv_kwh_interval / interval_hours
    grid_import_avoided_arbitrage_kw_orig[idx] = grid_import_avoided_arbitrage_kwh_interval / interval_hours
    battery_charge_kw_orig[idx] = (battery_charge_kwh_interval_grid + battery_charge_kwh_interval_pv) / interval_hours

t_orig = time.perf_counter() - t0
print(f"   ⏱️  Time: {t_orig:.3f}s")

# ============================================================================
# RUN VECTORIZED VERSION (Numba)
# ============================================================================
print("\n" + "=" * 80)
print("⚡ Step 2: Running VECTORIZED version (Numba JIT)")
print("=" * 80)

t0 = time.perf_counter()
results = _smart_battery_core(
    net_load_kw_np, net_load_kwh, price, future_price_low, future_price_high,
    future_peak_max_kw, future_excess_energy_kwh,
    peak_shaving_threshold_kw, battery_capacity_kwh, battery_power_kw,
    min_soc_kwh, other_uses_max_soc_kwh, interval_hours,
    peak_shaving_capacity_percent, peak_load, arbitrage_enabled,
    BATTERY_EFFICIENCY
)
t_vec = time.perf_counter() - t0

(battery_discharge_kw_vec, battery_discharge_ps_kw_vec, battery_discharge_ls_kw_vec,
 battery_soc_kwh_arr_vec, grid_import_kw_vec, grid_export_kw_vec, grid_export_pv_kw_vec,
 grid_import_avoided_arbitrage_kw_vec, battery_charge_kw_vec,
 arbitrage_charge_energy_kwh_vec, arbitrage_discharge_energy_kwh_vec,
 arbitrage_charge_prices_vec, arbitrage_discharge_prices_vec) = results

print(f"   ⏱️  Time: {t_vec:.3f}s")
print(f"\n🎉 SPEEDUP: {t_orig/t_vec:.1f}x FASTER!")

# ============================================================================
# VALIDATE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("✅ Step 3: Validating Results (CRITICAL)")
print("=" * 80)

arrays_to_check = [
    ("battery_discharge_kw", battery_discharge_kw_orig, battery_discharge_kw_vec),
    ("battery_discharge_ps_kw", battery_discharge_ps_kw_orig, battery_discharge_ps_kw_vec),
    ("battery_discharge_ls_kw", battery_discharge_ls_kw_orig, battery_discharge_ls_kw_vec),
    ("battery_soc_kwh", battery_soc_kwh_arr_orig, battery_soc_kwh_arr_vec),
    ("grid_import_kw", grid_import_kw_orig, grid_import_kw_vec),
    ("grid_export_kw", grid_export_kw_orig, grid_export_kw_vec),
    ("grid_export_pv_kw", grid_export_pv_kw_orig, grid_export_pv_kw_vec),
    ("grid_import_avoided_arbitrage_kw", grid_import_avoided_arbitrage_kw_orig, grid_import_avoided_arbitrage_kw_vec),
    ("battery_charge_kw", battery_charge_kw_orig, battery_charge_kw_vec),
    ("arbitrage_charge_energy_kwh", arbitrage_charge_energy_kwh_orig, arbitrage_charge_energy_kwh_vec),
    ("arbitrage_discharge_energy_kwh", arbitrage_discharge_energy_kwh_orig, arbitrage_discharge_energy_kwh_vec),
    ("arbitrage_charge_prices", arbitrage_charge_prices_orig, arbitrage_charge_prices_vec),
    ("arbitrage_discharge_prices", arbitrage_discharge_prices_orig, arbitrage_discharge_prices_vec),
]

all_match = True
for name, orig, vec in arrays_to_check:
    max_diff = np.abs(orig - vec).max()
    if max_diff < 1e-10:
        print(f"   ✓ {name:35s}: PERFECT MATCH (max diff: {max_diff:.2e})")
    else:
        print(f"   ✗ {name:35s}: MISMATCH (max diff: {max_diff:.2e})")
        all_match = False
        # Show where differences occur
        diff_indices = np.where(np.abs(orig - vec) > 1e-10)[0]
        print(f"      Differences at {len(diff_indices)} intervals: {diff_indices[:5]}...")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
if all_match:
    print("🎉 ✅ SUCCESS! VECTORIZED VERSION IS VALIDATED!")
    print("=" * 80)
    print(f"   • Accuracy: Perfect match on ALL {len(arrays_to_check)} arrays")
    print(f"   • Performance: {t_orig/t_vec:.1f}x faster ({t_orig*1000:.1f}ms → {t_vec*1000:.1f}ms)")
    print(f"   • Data size: {len(df_case_4):,} intervals")
    print("\n   ➜ run_battery_analysis() vectorization is PRODUCTION READY! 🚀")
else:
    print("❌ FAILED! Results don't match!")
    print("=" * 80)
    print("   ⚠️  DO NOT USE vectorized version in production!")
    sys.exit(1)

print("=" * 80)

