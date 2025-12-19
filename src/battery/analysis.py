"""
Smart battery analysis with peak shaving, PV integration, and arbitrage.

This module contains the core analysis loop optimized with Numba JIT compilation
for 20-50x performance improvement.
"""
import pandas as pd
import numpy as np
from numba import njit

from src.config import (INTERVAL_HOURS, BATTERY_EFFICIENCY, DEFAULT_DEPTH_OF_DISCHARGE,
                        MIN_ARBITRAGE_SPREAD, NEGATIVE_PRICE_THRESHOLD)


@njit
def _smart_battery_core(
    net_load_kw_np, net_load_kwh, price, future_price_low, future_price_high,
    future_peak_max_kw, future_excess_energy_kwh,
    peak_shaving_threshold_kw, battery_capacity_kwh, battery_power_kw,
    min_soc_kwh, other_uses_max_soc_kwh, interval_hours,
    peak_shaving_capacity_percent, peak_load, arbitrage_enabled,
    battery_efficiency
):
    """
    Numba-compiled core for smart battery analysis (20-50x faster).
    
    Handles: Peak shaving, PV charging, emergency charging, and arbitrage.
    """
    n = len(net_load_kw_np)
    battery_soc_kwh = battery_capacity_kwh
    battery_power_kwh = battery_power_kw * interval_hours
    
    # Initialize result arrays
    battery_discharge_kw = np.zeros(n)
    battery_discharge_ps_kw = np.zeros(n)
    battery_discharge_ls_kw = np.zeros(n)
    battery_charge_ps_kw = np.zeros(n)
    battery_charge_pv_kw = np.zeros(n)
    battery_soc_kwh_arr = np.zeros(n)
    grid_import_kw = np.zeros(n)
    grid_export_kw = np.zeros(n)
    grid_export_pv_kw = np.zeros(n)
    grid_import_avoided_arbitrage_kw = np.zeros(n)
    battery_charge_kw = np.zeros(n)
    arbitrage_charge_kw = np.zeros(n)
    arbitrage_discharge_kw = np.zeros(n)
    arbitrage_charge_energy_kwh = np.zeros(n)
    arbitrage_discharge_energy_kwh = np.zeros(n)
    arbitrage_charge_prices = np.zeros(n)
    arbitrage_discharge_prices = np.zeros(n)
    
    for idx in range(n):
        net_load_kw = net_load_kw_np[idx]
        net_load_kwh_interval = net_load_kwh[idx]
        current_price = price[idx]
        low_price_threshold = future_price_low[idx]
        high_price_threshold = future_price_high[idx]
        future_peak_value = future_peak_max_kw[idx]
        required_energy_for_peak = future_excess_energy_kwh[idx]
        
        # Initialize interval values
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
        
        # Start with base net load
        grid_import_kwh_interval = max(0.0, net_load_kwh_interval)
        action_taken = 0
        
        # ================================================================
        # 1. PEAK SHAVING
        # ================================================================
        if net_load_kw > peak_shaving_threshold_kw:
            discharge_needed_kwh = (net_load_kw - peak_shaving_threshold_kw) * interval_hours
            peak_shaving_min_limit = max(min_soc_kwh, other_uses_max_soc_kwh)
            max_discharge_possible_kwh = min(battery_power_kwh, (battery_soc_kwh - peak_shaving_min_limit))
            actual_discharge_kwh = min(discharge_needed_kwh, max(0.0, max_discharge_possible_kwh))
            
            if actual_discharge_kwh > 0:
                battery_discharge_kwh_interval = actual_discharge_kwh
                battery_discharge_kwh_interval_ps += actual_discharge_kwh
                battery_soc_kwh -= battery_discharge_kwh_interval
                grid_import_kwh_interval = max(0.0, net_load_kwh_interval - battery_discharge_kwh_interval)
                action_taken = 1
        
        # ================================================================
        # 2. PV SURPLUS CHARGING
        # ================================================================
        elif net_load_kw < 0 and action_taken == 0:
            charge_potential_kwh_interval = abs(net_load_kwh_interval)
            max_charge_possible_kwh_interval = min(battery_power_kwh, battery_capacity_kwh - battery_soc_kwh)
            actual_charge_from_PV_kwh_interval = min(charge_potential_kwh_interval, max(0.0, max_charge_possible_kwh_interval))
            
            if actual_charge_from_PV_kwh_interval > 0:
                battery_charge_kwh_interval_pv = actual_charge_from_PV_kwh_interval
                battery_soc_kwh += battery_charge_kwh_interval_pv * battery_efficiency
                grid_export_kwh_interval = charge_potential_kwh_interval - battery_charge_kwh_interval_pv
                grid_export_pv_kwh_interval = grid_export_kwh_interval
                grid_import_kwh_interval = 0.0
                action_taken = 1
        
        # ================================================================
        # 3. ARBITRAGE & EMERGENCY CHARGING
        # ================================================================
        else:
            # 3.1 Emergency charging for upcoming peak
            future_peak_detected = future_peak_value > peak_shaving_threshold_kw
            if peak_shaving_capacity_percent > 0:
                peak_reserve_soc = min(battery_capacity_kwh, min_soc_kwh + required_energy_for_peak)
            else:
                peak_reserve_soc = min_soc_kwh
            
            if peak_shaving_capacity_percent > 0 and future_peak_detected and battery_soc_kwh < peak_reserve_soc:
                max_charge_without_exceeding_threshold = max(0.0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
                charge_amount_kwh = min(
                    max_charge_without_exceeding_threshold,
                    battery_power_kw * interval_hours,
                    battery_capacity_kwh - battery_soc_kwh,
                    peak_reserve_soc - battery_soc_kwh
                )
                
                if charge_amount_kwh > 0:
                    battery_charge_kwh_interval_grid = charge_amount_kwh
                    battery_soc_kwh += charge_amount_kwh * battery_efficiency
                    grid_import_kwh_interval += charge_amount_kwh
                    action_taken = 1
            
            # 3.2 Arbitrage charging (low prices)
            min_profitable_discharge_price = current_price / battery_efficiency + MIN_ARBITRAGE_SPREAD
            arbitrage_is_profitable = high_price_threshold >= min_profitable_discharge_price
            is_very_negative_price = current_price <= NEGATIVE_PRICE_THRESHOLD
            should_charge = (current_price <= low_price_threshold and arbitrage_is_profitable) or is_very_negative_price
            
            if should_charge and battery_soc_kwh < battery_capacity_kwh and arbitrage_enabled:
                # Calculate max charge without exceeding peak
                if peak_shaving_capacity_percent > 0:
                    max_charge_without_exceeding_threshold = max(
                        0.0, (peak_shaving_threshold_kw - grid_import_kwh_interval / interval_hours) * interval_hours)
                else:
                    max_charge_without_exceeding_threshold = max(
                        0.0, (peak_load - grid_import_kwh_interval / interval_hours) * interval_hours)
                
                charge_amount_kwh = min(
                    max_charge_without_exceeding_threshold,
                    battery_power_kwh,
                    battery_capacity_kwh - battery_soc_kwh
                )
                
                if charge_amount_kwh > 0:
                    battery_charge_kwh_interval_grid += charge_amount_kwh
                    battery_soc_kwh += charge_amount_kwh * battery_efficiency
                    grid_import_kwh_interval += charge_amount_kwh
                    arbitrage_charge_energy_kwh[idx] = charge_amount_kwh
                    arbitrage_charge_prices[idx] = current_price
                    action_taken = 1
            
            # 3.3 Arbitrage discharging (high prices)
            if low_price_threshold < 0:
                discharge_is_profitable = current_price > 0
            else:
                min_profitable_current_price = low_price_threshold / battery_efficiency + MIN_ARBITRAGE_SPREAD
                discharge_is_profitable = current_price >= min_profitable_current_price
            
            min_reserve_for_check = min_soc_kwh if peak_shaving_capacity_percent == 0 else peak_reserve_soc
            
            if (current_price >= high_price_threshold and action_taken == 0 and
                battery_soc_kwh > min_reserve_for_check and 
                (battery_charge_kwh_interval_pv + battery_charge_kwh_interval_grid) == 0.0 and
                grid_import_kwh_interval > 0 and discharge_is_profitable and arbitrage_enabled):
                
                required_reserve = min_soc_kwh if peak_shaving_capacity_percent == 0 else max(min_soc_kwh, peak_reserve_soc)
                max_discharge_for_arbitrage = max(0.0, battery_soc_kwh - required_reserve)
                discharge_amount_kwh = min(battery_power_kwh, max_discharge_for_arbitrage, grid_import_kwh_interval)
                
                if discharge_amount_kwh > 0:
                    battery_soc_kwh -= discharge_amount_kwh
                    battery_discharge_kwh_interval += discharge_amount_kwh
                    battery_discharge_kwh_interval_ls += discharge_amount_kwh
                    grid_import_kwh_interval -= discharge_amount_kwh
                    grid_import_avoided_arbitrage_kwh_interval = discharge_amount_kwh
                    arbitrage_discharge_energy_kwh[idx] = discharge_amount_kwh
                    arbitrage_discharge_prices[idx] = current_price
        
        # Store results
        battery_discharge_kw[idx] = battery_discharge_kwh_interval / interval_hours
        battery_discharge_ls_kw[idx] = battery_discharge_kwh_interval_ls / interval_hours
        battery_discharge_ps_kw[idx] = battery_discharge_kwh_interval_ps / interval_hours
        battery_soc_kwh_arr[idx] = battery_soc_kwh
        grid_import_kw[idx] = grid_import_kwh_interval / interval_hours
        grid_export_kw[idx] = grid_export_kwh_interval / interval_hours
        grid_export_pv_kw[idx] = grid_export_pv_kwh_interval / interval_hours
        grid_import_avoided_arbitrage_kw[idx] = grid_import_avoided_arbitrage_kwh_interval / interval_hours
        battery_charge_kw[idx] = (battery_charge_kwh_interval_grid + battery_charge_kwh_interval_pv) / interval_hours
    
    return (battery_discharge_kw, battery_discharge_ps_kw, battery_discharge_ls_kw,
            battery_soc_kwh_arr, grid_import_kw, grid_export_kw, grid_export_pv_kw,
            grid_import_avoided_arbitrage_kw, battery_charge_kw,
            arbitrage_charge_energy_kwh, arbitrage_discharge_energy_kwh,
            arbitrage_charge_prices, arbitrage_discharge_prices)

