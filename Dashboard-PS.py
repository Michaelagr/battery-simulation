import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import locale
import datetime
#import test_daily_load

from src.config import (
    INTERVAL_HOURS,
    BATTERY_EFFICIENCY,
    DEFAULT_DEPTH_OF_DISCHARGE,
    VOLLLASTSTUNDEN_THRESHOLD,
    DEMAND_CHARGE_HIGH,
    DEMAND_CHARGE_LOW
)
from src.data.processors import handle_german_dst_transitions


# Streamlit config
st.set_page_config(page_title="Batteriesimulation Lastspitzenkappung", layout="wide", page_icon="💙")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1f77b4;}
    .metric {background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .plot-box {border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #fff; margin-bottom: 1rem;}
    .metric-title {font-size: 1.8rem; color: #444;font-weight: bold}
    .metric-value {font-size: 1.7rem; color: #111}
    </style>
""", unsafe_allow_html=True)

# Set German locale for number formatting
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'deu_deu')
    except locale.Error:
        st.write(" ")

# ADDED: This is the key change to prevent plots from opening in new tabs
plotly_config = {
    'displayModeBar': False,
    'showTips': False,
    'displaylogo': False,
    'scrollZoom': True,
    'staticPlot': False
}
#from graphs import demand_charge



# .metric-value {font-size: 1.5rem; color: #111; font-weight: bold;}
st.title("🔋 ecoplanet Batterie-Simulation für Lastspitzenkappung")

#--------------------- Helper Definitions -------------------------
# Initial battery configuration
battery_efficiency = BATTERY_EFFICIENCY
discharge_percentage = 0.001
demand_charge = 200
template_url = "https://docs.google.com/spreadsheets/d/1xJ3Lk8uy3X8piSt-IUxZgeVYqeOuRG9N/edit?usp=sharing&ouid=114799245841423325825&rtpof=true&sd=true"
battery_configs = {
    "Small": {"capacity": 90, "power": 92},
    "Medium": {"capacity": 215, "power": 100},
    "Large": {"capacity": 500, "power": 250}
}

# -------------------- Helper functions --------------------------

### Peak Shaving Simulation
def peak_shaving(load_data, threshold):
    peak_threshold = max(load_data) * (threshold / 100)
    optimized_load = np.where(load_data > peak_threshold, peak_threshold, load_data)
    return optimized_load

####
def battery_simulation_ps(df, battery_capacity, power_rating, threshold_kw, depth_of_discharge, battery_efficiency):
    #"""
    #Peak shaving battery simulation function
    #"""
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 10%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = INTERVAL_HOURS

    threshold_kw = df["net_load_kw"].max() - threshold_kw

    optimized = []
    charge = []
    discharge = []
    soc_state = []

    for load in df["net_load_kw"]:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # energy_used = actual_discharge_power * interval_hours / 1
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours
            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            charge.append(actual_charge_power)
            discharge.append(0)

        else:
            charge.append(0)
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["ps_grid_load"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df



### New battery simulation
def battery_simulation_v02(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 20%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = INTERVAL_HOURS
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    charge = []
    discharge = []
    soc_state = []

    for load in df["load"]:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # soc = max(soc - energy_used, reserve_energy)
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours
            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            charge.append(actual_charge_power)
            discharge.append(0)

        else:
            charge.append(0)
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df


def battery_simulation_hlz_only(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY):
    """
    Battery simulation that only operates during HLZ (Hochlastzeitfenster) windows.
    The battery only charges/discharges when 'in_window' is True.
    """
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = INTERVAL_HOURS
    
    # Calculate threshold based on peak in HLZ windows only
    hlz_data = df[df['in_window']]
    if len(hlz_data) > 0:
        peak_hlz = hlz_data["load"].max()
        threshold_kw = peak_hlz * (threshold_pct / 100)
    else:
        threshold_kw = df["load"].max() * (threshold_pct / 100)  # Fallback

    optimized = []
    charge = []
    discharge = []
    soc_state = []

    for i, (load, in_window) in enumerate(zip(df["load"], df["in_window"])):
        grid_load = load  # start with original load

        # Only operate battery during HLZ windows
        if in_window:
            # --- DISCHARGING ---
            if load > threshold_kw and soc > reserve_energy:
                power_needed = load - threshold_kw
                max_discharge_power = (soc - reserve_energy) / interval_hours
                actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

                energy_used = actual_discharge_power * interval_hours / battery_efficiency
                soc = soc - energy_used

                grid_load = load - actual_discharge_power
                charge.append(0)
                discharge.append(actual_discharge_power)

            # --- CHARGING (only when load is below threshold to avoid peak increase) ---
            elif load <= threshold_kw and soc < total_capacity:
                max_possible_charge = threshold_kw - load  # Max charge without exceeding threshold

                max_charge_power = (total_capacity - soc) / interval_hours
                actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

                energy_stored = actual_charge_power * interval_hours * battery_efficiency
                soc = min(soc + energy_stored, total_capacity)

                grid_load = load + actual_charge_power
                charge.append(actual_charge_power)
                discharge.append(0)

            else:
                charge.append(0)
                discharge.append(0)
        else:
            # Outside HLZ windows: battery is inactive
            charge.append(0)
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df




### New battery simulation
def battery_simulation_vpv(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY):
    # Idee: Peak shifting: Ab ~Juni: 50 % der Kapazität für Load shifting reservieren
        # Zähler n=0, if n>15*24*30*5 ~~20.000:
            # total_capacity = battery_capacity if or(n<20.000, n>40000) else total_capacity =  capactiy_sommer = 0.5* total capacity
            # sommer_capacity = 0.5*total_capacity
    # Which columns do I need to properly simulate the battery - for peakshaving discharge & charge, peak shifting discharge & charge, total charge & discharge
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = INTERVAL_HOURS
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            charge_pv.append(0)
            charge_grid.append(0)

        # --- CHARGING ---
        else:
            # 1. Charge from PV first (always allowed, up to available PV and battery limits)
            max_charge_power = (total_capacity - soc) / interval_hours  # what battery can take
            # FALLS es PV gibt - also pv_neg <0 ist - dann mit pv laden, maximal so viel wie geht. Sonst is pv_chaege_power 0
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            soc_available = total_capacity - soc
            if soc_available > 0 and load + pv_charge_power < threshold_kw:
                # How much more can be charged without exceeding threshold?
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(power_rating - pv_charge_power, max_grid_charge_power, soc_available / interval_hours)
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(0)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    return df

def battery_simulation_vpv_selfconsumption_oldf(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = INTERVAL_HOURS
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(6, 10)

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            # Peak shaving as before
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # Default all PV charging to self-consumption, unless peak shaving is triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            soc_available = total_capacity - soc
            if soc_available > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(power_rating - pv_charge_power, max_grid_charge_power, soc_available / interval_hours)
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    battery_charge_grid_self = grid_charge_power  # Default all grid charging to self-consumption, unless peak shaving is triggered
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power

            # --- PV-rich period: Discharge for self-consumption even if below threshold ---
            if row['pv_rich'] and load > 0 and soc > reserve_energy:
                discharge_power = min(load, power_rating, (soc - reserve_energy) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, reserve_energy), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    return df


def optimize_battery_params(df, battery_capacity, power_rating, demand_charge_low, demand_charge_high, energy_charge_low, energy_charge_high):
    best_roi = -float('inf')
    best_params = {}

    # Define parameter ranges
    threshold_range = range(75, 95, 5)  # 60-90% in 5% steps
    base_reserve_range = [0.2, 0.3, 0.4]
    extra_reserve_range = [0.4, 0.5]

    for threshold in threshold_range:
        for base_reserve in base_reserve_range:
            for extra_reserve in extra_reserve_range:
                # Run simulation
                result_df = battery_simulation_vpv_selfconsumption(
                    df.copy(), battery_capacity, power_rating,
                    DEFAULT_DEPTH_OF_DISCHARGE, threshold, base_reserve=base_reserve,
                    extra_reserve=extra_reserve
                )

                vlh = df["grid_load_pv_bt"].sum() /4 / df["grid_load_pv_bt"].max()
                demand_charge_calc = demand_charge_low if vlh < VOLLLASTSTUNDEN_THRESHOLD else demand_charge_high
                energy_charge_calc = energy_charge_high if vlh < VOLLLASTSTUNDEN_THRESHOLD else energy_charge_low

                # Calculate ROI
                roi = calculate_roi(result_df, demand_charge_calc, energy_charge_calc)

                if roi > best_roi:
                    best_roi = roi
                    best_params = {
                        'threshold_pct': threshold,
                        'base_reserve': base_reserve,
                        'extra_reserve': extra_reserve,
                        'roi': roi
                    }

    return best_params

def calculate_roi(df, demand_charge, energy_charge):
    savings_ps = (df["load_pv"].max() - df["grid_load_pv_bt"].max()) * demand_charge
    pv_selfcons_kwh = min(
        df['battery_charge_pv_selfcons'].sum() / 4,
        df['battery_discharge_selfcons'].sum() / 4
    )
    annual_savings_selfcons = pv_selfcons_kwh * energy_charge
    savings_total = savings_ps + annual_savings_selfcons

    return savings_total


def optimize_battery_params_working(df, battery_capacity, power_rating, demand_charge_low, demand_charge_high, energy_charge_low, energy_charge_high):
    best_roi = -float('inf')
    best_params = {}

    # Define parameter ranges
    threshold_range = range(75, 95, 5)  # 60-90% in 5% steps
    reserve_fraction_range = [0.2, 0.3, 0.4]

    for threshold in threshold_range:
        for reserve_fraction in reserve_fraction_range:
            # Run simulation
            result_df = battery_simulation_vpv_selfconsumption_working(
                df.copy(), battery_capacity, power_rating,
                DEFAULT_DEPTH_OF_DISCHARGE, threshold,
                reserve_fraction = reserve_fraction
            )

            vlh = df["grid_load_pv_bt"].sum() / df["grid_load_pv_bt"].max()
            demand_charge_calc = demand_charge_low if vlh < VOLLLASTSTUNDEN_THRESHOLD else demand_charge_high
            energy_charge_calc = energy_charge_high if vlh < VOLLLASTSTUNDEN_THRESHOLD else energy_charge_low

            # Calculate ROI
            roi = calculate_roi(result_df, demand_charge_calc, energy_charge_calc)

            if roi > best_roi:
                best_roi = roi
                best_params = {
                    'threshold_pct': threshold,
                    'reserve_fraction': reserve_fraction,
                    'roi': roi
                }

    return best_params


def battery_simulation_vpv_selfconsumption_working(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY, reserve_fraction=0.1):
    ################### ELAS CHANGES ###############
    max_load_jump = df["load"].diff().clip(lower=0).max()
    #reserve_fraction = max_load_jump / df["load"].max()
    #reserve_fraction = max_load_jump /4 / battery_capacity
    reserve_amount = max_load_jump /4

    ##########################
    total_capacity = battery_capacity  # kWh
    true_min_soc = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    reserve_soc = true_min_soc + (total_capacity - true_min_soc) * reserve_fraction
    #    reserve_soc = true_min_soc + reserve_amount

    soc = total_capacity  # start fully charged
    interval_hours = INTERVAL_HOURS
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(start_month_pv, end_month_pv)
    #df['pv_rich'] = df['month'].isin([1,2,3,4, 5, 6, 8, 9,12]) // LB Profil

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []
    soc_reserve_state = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv = row["pv"]
        pv_neg = row["load_pv_neg"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # --- DISCHARGING ---
        if load > threshold_kw and soc > true_min_soc:
            # Peak shaving: allow discharge down to technical minimum
            power_needed = load - threshold_kw
            max_discharge_power = (soc - true_min_soc) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first (up to available space)
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # All PV charging is for self-consumption unless peak shaving triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            # --- CHANGED LOGIC HERE ---
            if row['pv_rich']:
                # In PV-rich (summer): only fill grid up to reserve
                soc_available_for_grid = reserve_soc - soc
            else:
                # In winter: allow grid charging up to 100%
                soc_available_for_grid = total_capacity - soc
            # -------------------------
            if soc_available_for_grid > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(
                    power_rating - pv_charge_power,
                    max_grid_charge_power,
                    soc_available_for_grid / interval_hours
                )
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    if row['pv_rich']:
                        battery_charge_grid_self = 0  # grid charging is only for reserve (peak shaving), not self-consumption
                        battery_charge_grid_peak = grid_charge_power
                    else:
                        battery_charge_grid_self = 0  # in winter, all grid charging is for peak shaving
                        battery_charge_grid_peak = grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
                    battery_charge_grid_self = 0
                    battery_charge_grid_peak = 0
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power
                battery_charge_grid_self = 0
                battery_charge_grid_peak = 0

            # --- PV-rich period: Discharge for self-consumption down to reserve_soc (not technical min) ---
            if row['pv_rich'] and load > 0 and soc > reserve_soc:
                # Only discharge self-consumption down to reserve_soc
                discharge_power = min(load, power_rating, (soc - reserve_soc) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, true_min_soc), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(battery_charge_grid_peak)

        optimized.append(grid_load)
        soc_state.append(soc)
        soc_reserve_state.append(reserve_soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    df["soc_reserve"] = soc_reserve_state
    return df


def battery_simulation_vpv_selfconsumption(
    df,
    battery_capacity,
    power_rating,
    depth_of_discharge,
    threshold_pct,
    battery_efficiency=BATTERY_EFFICIENCY,
    base_reserve=0.3,
    extra_reserve=0.,
    reserve_window_hours=24
    ):
    total_capacity = battery_capacity  # kWh
    true_min_soc = total_capacity * (1 - depth_of_discharge / 100)  # min SoC in kWh
    soc = total_capacity  # start fully charged
    interval_hours = INTERVAL_HOURS

    # Calculate threshold for peak shaving
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    # Add PV-rich flag (June to October)
    df['month'] = pd.to_datetime(df['timestamp']).dt.month
    df['pv_rich'] = df['month'].between(2, 10)

    # Calculate the largest positive jump in net load (after PV)
    max_load_jump = df["load"].diff().clip(lower=0).max()
    jump_reserve = max_load_jump * interval_hours
    min_reserve_fraction = 0.2

    # Calculate rolling peak and dynamic reserve fraction
    window_intervals = int(reserve_window_hours / interval_hours)
    df['rolling_peak'] = df['load'].rolling(window=window_intervals, min_periods=1).max()
    max_peak = df['load'].max()
    df['reserve_fraction_dynamic'] = base_reserve + extra_reserve * (df['rolling_peak'] / max_peak)
    df['reserve_fraction_dynamic'] = df['reserve_fraction_dynamic'].clip(upper=0.9)

    # Tracking lists
    optimized = []
    charge = []
    discharge = []
    soc_state = []
    charge_pv = []
    charge_grid = []
    discharge_peakshave = []
    discharge_selfcons = []
    charge_pv_selfcons = []
    charge_pv_peakshave = []
    charge_grid_selfcons = []
    charge_grid_peakshave = []
    soc_reserve_state = []

    for idx, row in df.iterrows():
        load = row["load"]
        pv_neg = row["load_pv_neg"]
        #pv = row["pv"]
        grid_load = load  # start with original load

        battery_charge_power_pv = 0
        battery_charge_power_grid = 0
        battery_discharge_peakshave = 0
        battery_discharge_selfcons = 0
        battery_charge_pv_self = 0
        battery_charge_pv_peak = 0
        battery_charge_grid_self = 0
        battery_charge_grid_peak = 0

        # Dynamic reserve for this timestep
        reserve_fraction = row['reserve_fraction_dynamic']
        #reserve_soc = true_min_soc + (total_capacity - true_min_soc) * reserve_fraction

        reserve_soc = max(true_min_soc + jump_reserve,
                          true_min_soc + (total_capacity - true_min_soc) * min_reserve_fraction,
                          true_min_soc + (total_capacity - true_min_soc) * reserve_fraction)


        # --- DISCHARGING: Peak shaving ---
        if load > threshold_kw and soc > true_min_soc:
            # Peak shaving: allow discharge down to technical minimum
            power_needed = load - threshold_kw
            max_discharge_power = (soc - true_min_soc) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power, load)
            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            soc -= energy_used
            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)
            discharge_peakshave.append(actual_discharge_power)
            discharge_selfcons.append(0)
            charge_pv.append(0)
            charge_grid.append(0)
            charge_pv_selfcons.append(0)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(0)
            charge_grid_peakshave.append(0)

        else:
            # 1. Charge from PV first (up to available space)
            max_charge_power = (total_capacity - soc) / interval_hours
            pv_charge_power = min(power_rating, max_charge_power, -pv_neg) if pv_neg < 0 else 0
            energy_stored_pv = pv_charge_power * interval_hours * battery_efficiency
            soc += energy_stored_pv
            battery_charge_power_pv = pv_charge_power
            battery_charge_pv_self = pv_charge_power  # All PV charging is for self-consumption unless peak shaving triggered

            # 2. If battery not full, charge from grid (only if grid load stays below threshold)
            if row['pv_rich']:
                soc_available_for_grid = reserve_soc - soc
            else:
                soc_available_for_grid = total_capacity - soc

            if soc_available_for_grid > 0 and load + pv_charge_power < threshold_kw:
                max_grid_charge_power = threshold_kw - (load + pv_charge_power)
                grid_charge_power = min(
                    power_rating - pv_charge_power,
                    max_grid_charge_power,
                    soc_available_for_grid / interval_hours
                )
                if grid_charge_power > 0:
                    energy_stored_grid = grid_charge_power * interval_hours * battery_efficiency
                    soc += energy_stored_grid
                    battery_charge_power_grid = grid_charge_power
                    grid_load = load + pv_charge_power + grid_charge_power
                    if row['pv_rich']:
                        battery_charge_grid_self = 0  # grid charging is only for reserve (peak shaving), not self-consumption
                        battery_charge_grid_peak = grid_charge_power
                    else:
                        battery_charge_grid_self = 0  # in winter, all grid charging is for peak shaving
                        battery_charge_grid_peak = grid_charge_power
                else:
                    battery_charge_power_grid = 0
                    grid_load = load + pv_charge_power
                    battery_charge_grid_self = 0
                    battery_charge_grid_peak = 0
            else:
                battery_charge_power_grid = 0
                grid_load = load + pv_charge_power
                battery_charge_grid_self = 0
                battery_charge_grid_peak = 0

            # --- PV-rich period: Discharge for self-consumption down to reserve_soc (not technical min) ---
            if row['pv_rich'] and load > 0 and soc > reserve_soc:
                # Only discharge self-consumption down to reserve_soc
                discharge_power = min(load, power_rating, (soc - reserve_soc) / interval_hours)
                energy_used = discharge_power * interval_hours / battery_efficiency
                soc -= energy_used
                grid_load -= discharge_power
                battery_discharge_selfcons = discharge_power
            else:
                battery_discharge_selfcons = 0

            # Clamp SoC to valid range
            soc = min(max(soc, true_min_soc), total_capacity)

            charge.append(battery_charge_power_pv + battery_charge_power_grid)
            discharge.append(battery_discharge_selfcons)
            discharge_peakshave.append(0)
            discharge_selfcons.append(battery_discharge_selfcons)
            charge_pv.append(battery_charge_power_pv)
            charge_grid.append(battery_charge_power_grid)
            charge_pv_selfcons.append(battery_charge_pv_self)
            charge_pv_peakshave.append(0)
            charge_grid_selfcons.append(battery_charge_grid_self)
            charge_grid_peakshave.append(battery_charge_grid_peak)

        optimized.append(grid_load)
        soc_state.append(soc)
        soc_reserve_state.append(reserve_soc)

    # Results columns
    df["grid_load_pv_bt"] = optimized
    df["battery_charge"] = charge
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    df["battery_charge_pv"] = charge_pv
    df["battery_charge_grid"] = charge_grid
    df["battery_discharge_peakshave"] = discharge_peakshave
    df["battery_discharge_selfcons"] = discharge_selfcons
    df["battery_charge_pv_selfcons"] = charge_pv_selfcons
    df["battery_charge_pv_peakshave"] = charge_pv_peakshave
    df["battery_charge_grid_selfcons"] = charge_grid_selfcons
    df["battery_charge_grid_peakshave"] = charge_grid_peakshave
    df["pv_rich"] = df['pv_rich']
    df["soc_reserve"] = soc_reserve_state
    df["reserve_fraction_dynamic"] = df['reserve_fraction_dynamic']

    return df


def battery_simulation_ps_with_pv(df, load_series, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=BATTERY_EFFICIENCY):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 20%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = INTERVAL_HOURS
    peak = load_series.max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    discharge = []
    soc_state = []
    battery_load_from_pv = []

    for load in load_series:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # soc = max(soc - energy_used, reserve_energy)
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours

            #if

            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            discharge.append(0)

        else:
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load_pv_bt"] = optimized
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df


#--------------------------------------------- FILE PROCESSING ------------------------------------------------------
peak_load_pv = 0
energy_consumed_pv = 0
volllaststunden_pv = 0
pv_energy_exported = 0
pv_generated = 0
pv_self_consumed = 0
pv_self_consumption_ratio = 0
autarky_ratio = 0



############################################# File Processing #############################################
######### Uploader #########
with st.sidebar:
            # Logo in sidebar
    try:
        st.image("ecoplanet_logo.png", width=150)
        st.markdown("---")
    except:
        pass  # Logo file not found, continue without it

    
    #st.subheader("Datei-Auswahl")
    #st.write("Find a template [here](%s)" % template_url)
    # Option to upload file or select from existing files
    st.write("**📁 Lastgang:**")
    

    
    # Get list of files from /input folder
    import os
    input_folder = "input/demo"
    existing_files = []
    
    if os.path.exists(input_folder):
        existing_files = [f for f in os.listdir(input_folder) if f.endswith('.xlsx')]
    
    uploaded_file = None
    selected_file = "Keine Auswahl"
    
    if existing_files:
        selected_file = st.selectbox("📂 Vorhandene Datei auswählen", ["Keine Auswahl"] + existing_files)
    else:
        st.info("Keine .xlsx Dateien im /input Ordner gefunden.")

    # Option to upload
    st.write("**oder**")

    with st.expander("📤 Neue Datei (xlsx)", expanded=False):
    # File upload option
        uploaded_file_from_uploader = st.file_uploader("📤 Neue Datei hochladen (xlsx)", type=["xlsx"], help="Datei muss zwei Spalten enthalten: 'timestamp' und 'load'")
    
    # Determine which file to use: prioritize dropdown selection over uploaded file
    if selected_file != "Keine Auswahl":
        # Use selected file from dropdown
        file_path = os.path.join(input_folder, selected_file)
        uploaded_file = file_path
    elif uploaded_file_from_uploader is not None:
        # Use uploaded file if no dropdown selection
        uploaded_file = uploaded_file_from_uploader
    
    st.write("---")
    st.write("**☀️ PV-Profil:**")
    pv_total = st.number_input("☀️ PV (kw peak)", min_value=-10000000, value=0)
    st.write("**oder**")
    with st.expander("📤 Eigenes PV-Profil hochladen (xlsx)", expanded=False):
    # Custom PV profile upload option
        st.write("**☀️ PV-Profil:**")
        custom_pv_file = st.file_uploader("📤 Eigenes PV-Profil hochladen (xlsx)", type=["xlsx"], 
                                        help="Datei muss zwei Spalten enthalten: 'timestamp' und 'PV' (in kW)")
    if custom_pv_file:
        st.info("✅ Eigenes PV-Profil wird verwendet")
    elif pv_total > 0:
        st.info("📊 Standard-PV-Profil wird verwendet")
    
    st.write("---")
    st.write(f"**Netznutzungsentgelte**")
    st.write(f"**<{VOLLLASTSTUNDEN_THRESHOLD} VLH**")
    demand_charge_low = st.number_input(f"💰 Leistungspreis <{VOLLLASTSTUNDEN_THRESHOLD} VLH (in €/kW)", min_value=None, value=DEMAND_CHARGE_LOW)
    energy_charge_high_input = st.number_input(f"💲 Arbeitspreis <{VOLLLASTSTUNDEN_THRESHOLD} VLH (in ct/kWh)",min_value=None, value=10.00)
    st.write(f"**>={VOLLLASTSTUNDEN_THRESHOLD} VLH**")
    demand_charge_high = st.number_input(f"💰️ Leistungspreis >={VOLLLASTSTUNDEN_THRESHOLD} VLH (in €/kW)", min_value=None, value=DEMAND_CHARGE_HIGH)
    energy_charge_low_input = st.number_input(f"💲 Arbeitspreis >={VOLLLASTSTUNDEN_THRESHOLD} VLH (in ct/kWh)", min_value=None, value= 1.00)

    energy_charge_high = energy_charge_high_input /100
    energy_charge_low = energy_charge_low_input /100

    ## Handling
    if uploaded_file:
        # Handle both file path (string) and uploaded file object
        if isinstance(uploaded_file, str):
            # File selected from input folder
            df = pd.read_csv(uploaded_file) if uploaded_file.endswith(".csv") else pd.read_excel(uploaded_file)
        else:
            # File uploaded via file uploader - need to find header row
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        
        # Try to auto-detect datetime columns
        col_map = {col.lower(): col for col in df.columns}
        date_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["date", "day", "tag"])), None)
        time_col = next((col for key, col in col_map.items() if any(
            kw in key for kw in ["time", "hour", "timestamp", "zeit", "uhrzeit", "timestamps", "datum"])), None)
        load_col = next((col for key, col in col_map.items() if any(
            kw in key for kw in ["kw", "load", "value", "value_kw", "power", "entnahme", "last", "netzbezug", "wert", "messung"])), None)

        if date_col and time_col:
            df["timestamp"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), format='%d.%m.%Y %H:%M')

        elif time_col:
            # st.sidebar.write(f"**{time_col}**")
            try:
                df["timestamp"] = pd.to_datetime(df[time_col], dayfirst=True)
            except Exception as e1:
                try:
                    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)
                except Exception as e2:
                    try:
                        # st.sidebar.write(f"**Do we even get here?{time_col}**")
                        df["timestamp"] = pd.to_datetime(df[time_col], format="mixed", dayfirst=True)
                    except Exception as e3:
                        try:
                            # st.sidebar.write(f"**Do we even get here?{time_col}**")
                            # needs to be ok with   2024-01-01T00:15:00+01:00 or 2024-01-24T10:00:00+01:00
                            df["timestamp"] = pd.to_datetime(df[time_col], format="%Y-%m-%dT%H:%M:%S%z")
                        except Exception as e4:
                            try:
                                df["timestamp"] = pd.to_datetime(df[time_col], format="ISO8601")
                                # ['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                            except Exception as e5:
                                raise ValueError(
                                    f"Datei konnte nicht gelesen werden. Fehler:\n1. {e1}\n2. {e2}\n3. {e3}\n4. {e4}\n5. {e5}")
        elif date_col:
            try:
                df["timestamp"] = pd.to_datetime(df[date_col], dayfirst=True)
            except Exception as e6:
                raise ValueError(f"Datei konnte nicht gelesen werden.")
        else:
            df["timestamp"] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)  # fallback

        if load_col:
            if (load_col == "value_kwh") or (load_col == "kWh"):
                df["load"] = pd.to_numeric(df[load_col], errors='coerce') * 4
            else:
                df["load"] = pd.to_numeric(df[load_col], errors='coerce')

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        
        # Handle German DST transitions after timestamp parsing
        df = handle_german_dst_transitions(df)
        
        # Clean and validate load data early
        if 'load' in df.columns:
            df["load"] = pd.to_numeric(df["load"], errors='coerce')
            initial_len = len(df)
            df = df.dropna(subset=['load'])
            if len(df) < initial_len:
                st.warning(f"⚠️ {initial_len - len(df)} Zeilen mit ungültigen Lastdaten entfernt.")
            
            if len(df) == 0:
                st.error("❌ Keine gültigen Lastdaten nach der Bereinigung gefunden.")
                st.stop()
        
        df = df.set_index("timestamp", drop=False)

        # Create date columns
        df["datetime"] = df["timestamp"]
        df["Date"] = df["timestamp"].dt.date
        #    df["week"] = df["timestamp"].dt.strftime("%G-W%V-%u")
        df["Week"] = df["timestamp"].dt.isocalendar().week
        df["Month"] = df["timestamp"].dt.strftime("%Y-%m")
        df["Year"] = df["timestamp"].dt.strftime("%Y")
        df["Weekday"] = df["timestamp"].dt.day_name()
        df["time"] = df["timestamp"].dt.time




        ########### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ READ IN SOLAR DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ### -------------------------------------------------- PV ------------------------------------------------------------ ####
        MAGIC_YEARLY_PV_MULTIPLIER = 800

        # INTERVAL_HOURS imported from src.config
        
        # Check if custom PV file is uploaded
        if custom_pv_file is not None:
            # Load custom PV profile
            df_pv = pd.read_excel(custom_pv_file)
            
            # Find timestamp and PV columns
            pv_timestamp_col = None
            pv_load_col = None
            
            for col in df_pv.columns:
                col_lower = str(col).lower()
                if 'timestamp' in col_lower:
                    pv_timestamp_col = col
                elif 'pv' in col_lower:
                    pv_load_col = col
            
            if pv_timestamp_col is None or pv_load_col is None:
                st.error("❌ PV-Datei muss Spalten 'timestamp' und 'PV' enthalten!")
                st.stop()
            
            # Process custom PV data
            try:
                # Try with dayfirst=True for European date formats (DD.MM.YYYY)
                df_pv["timestamp"] = pd.to_datetime(df_pv[pv_timestamp_col], dayfirst=True).dt.tz_localize(None)
            except ValueError:
                try:
                    # Fallback: let pandas infer the format
                    df_pv["timestamp"] = pd.to_datetime(df_pv[pv_timestamp_col], format='mixed').dt.tz_localize(None)
                except ValueError:
                    st.error("❌ Fehler beim Parsen der Zeitstempel im PV-Profil. Bitte verwenden Sie das Format DD.MM.YYYY HH:MM")
                    st.stop()
            
            df_pv["yearly_production_kw"] = pd.to_numeric(df_pv[pv_load_col], errors='coerce')
            df_pv["yearly_production_kwh"] = df_pv["yearly_production_kw"] * INTERVAL_HOURS
            
        else:
            # Use standard PV profile
            df_pv = pd.read_csv("data/solar_data_de_small.csv")
            
            # Dataframe to store PV distribution
            df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], format="%d.%m.%y %H:%M")
            df_pv["yearly_production_kw"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(
                min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER / INTERVAL_HOURS  # result in kW
            df_pv["yearly_production_kwh"] = df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(
                min=0) * pv_total * MAGIC_YEARLY_PV_MULTIPLIER  # result in kwh
            
        df_pv.index = pd.to_datetime(df_pv.index).tz_localize(None)

        # Create load with PV in "load_pv"
        df["timestamp_index"] = df["timestamp"]
        df_pv["timestamp_index"] = df_pv["timestamp"]
        
        # Remove duplicates before setting index (DST already handled earlier)
        df = df[~df['timestamp_index'].duplicated(keep='first')]
        df_pv = df_pv[~df_pv['timestamp_index'].duplicated(keep='first')]
        
        df = df.set_index('timestamp_index')
        df_pv = df_pv.set_index('timestamp_index')
        df["pv"] = df_pv["yearly_production_kw"]
        df["load_org"] = df["load"]
        df["load_pv"] = df["load"].sub(df_pv["yearly_production_kw"], fill_value=0)

        # Positive Werte (Netzbezug) und negative Werte (Einspeisung) berücksichtigen
        df["load_pv_pos"] = df["load_pv"].clip(lower=0)
        df["load_pv_neg"] = df["load_pv"].clip(upper=0)
        positive_load_pv = df["load_pv"][df["load_pv"] > 0]
        negative_load_pv = df["load_pv"][df["load_pv"] < 0]

        # df ist original - df_with_pv ist mit PV
        df_with_pv = df.copy()
        if pv_total > 0 or custom_pv_file is not None:
            df_with_pv["load"] = df["load_pv"]
        else:
            df_with_pv["load"] = df["load"]



        with st.expander ("### 🔎 Raw data preview"):
            st.write(df.head())

    # -------------------------- PV ---------------------------------------




################################################################################ LOAD PROFILE INFORMATION ######################################################

tab_analyse, taboptimierer, tabsimulation, = st.tabs(["📈 Lastgang Analyse", "⭐ Batterie-Optimierer", "🔋 Batterie-Simulation"])

with ((tab_analyse)):
    if not uploaded_file:
        with st.container(border = True):
            st.subheader("⚠️ Bitte laden Sie ihren Lastgang in der Dokumenten-Auswahl auf der linken Seite hoch.")
            #st.write("**Ensure that the file contains two columns with a timestamp and the load in kW.**")
            #st.write("**In case you encounter an error, ensure that the naming of the columns is 'timestamp' and 'load'.**")
            #st.write("A template can be downloaded [here](%s)" % template_url)
            #st.write("Please note: The ideal format for the timestamp is dd.mm.yyyy hh:mm")

    if uploaded_file:


        df_org = df.copy()


        ## ------------------------------------------    VARIABLEN    --------------------------------------------
        total_entries = len(df)
        # Clean load data - remove NaN values and ensure numeric
        df["load"] = pd.to_numeric(df["load"], errors='coerce')
        df = df.dropna(subset=['load'])
        
        if len(df) == 0:
            st.error("❌ Keine gültigen Lastdaten gefunden. Bitte überprüfen Sie Ihre Datei.")
            st.stop()
        
        avg_load = df["load"].mean()
        min_load = df["load"].min()
        peak_load = df["load"].max()
        
        # Additional validation
        if pd.isna(peak_load) or peak_load <= 0:
            st.error("❌ Ungültige Spitzenlast gefunden. Bitte überprüfen Sie Ihre Lastdaten.")
            st.stop()
            
        total_energy_kwh = df["load"].sum() /4  # assuming 15-min intervals
        total_energy_Mwh = df["load"].sum() / 4 / 1000  # assuming 15-min intervals
        volllaststunden = df["load"].sum() / 4 / peak_load

        # PV METRICS
        peak_load_pv = df_with_pv["load"].max()
        energy_consumed_pv = positive_load_pv.sum() / 4 / 1000
        volllaststunden_pv = positive_load_pv.sum() / 4 / positive_load_pv.max()
        pv_energy_exported = -df["load_pv_neg"].sum() * INTERVAL_HOURS  # Eingespeiste Energie (PV-Überschuss)
        pv_generated = df_pv["yearly_production_kw"].sum() * INTERVAL_HOURS  # PV-Daten (gesamte PV-Erzeugung)
        pv_self_consumed = pv_generated - pv_energy_exported  # Eigenverbrauch PV (direkt genutzte PV-Energie)
        pv_self_consumption_ratio = pv_self_consumed / pv_generated * 100  if (pv_total > 0 or custom_pv_file is not None) and pv_generated > 0 else 0# Eigenverbrauchsquote
        autarky_ratio = pv_self_consumed / total_energy_kwh * 100  # Autarkiegrad

        ## -----------------------------------------------------------------------------------------------------

        st.header("Lastgang Übersicht")

        st.subheader("📈 Statistik")
        st.write(f"Von **{df['timestamp'].min()}** bis **{df['timestamp'].max()}**")

        col1, col2, col3, col4, col5 = st.columns(5)


        #col1.markdown("<div class='metric'><div class='metric-title'> 📅 Total datapoints </div><div class='metric-value'>" + f"{total_entries:,.0f}" + "</div></div>",unsafe_allow_html=True)
        col1.markdown(
            "<div class='metric'><div class='metric-title'> 🔌 Gesamtverbrauch </div><div class='metric-value'>" + f"{total_energy_Mwh:,.0f} MWh" + "</div></div>",
            unsafe_allow_html=True)
        col2.markdown(
            "<div class='metric'><div class='metric-title'> Ø  Durchschn. Last </div><div class='metric-value'>" + f"{avg_load:,.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col3.markdown(
            "<div class='metric'><div class='metric-title'> 🔺 Max. Last </div><div class='metric-value'>" + f"{peak_load:,.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col4.markdown(
            "<div class='metric'><div class='metric-title'> 🔻 Min. Last </div><div class='metric-value'>" + f"{min_load:.1f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col5.markdown(
            "<div class='metric'><div class='metric-title'> ⏺️ Volllaststunden </div><div class='metric-value'>" + f"{(total_energy_kwh / peak_load ):.0f} h" + "</div></div>",
            unsafe_allow_html=True)


        st.write("\n")

        tab_param, tab_agg, tab_solar, tab_toppeaks, tab_loadduration  = st.tabs(["Parameter", "Aggregierte Ansicht", "PV Analyse", "Lastspitzen-Analyse", "Lastdauerkurve"])

        with tab_param:
            col1, col2 = st.columns([1,2])
            with col1:
                with st.container(border=True):
                    st.subheader("💶 Netznutzungsentgelte")
                    st.write(f"**<{VOLLLASTSTUNDEN_THRESHOLD} Vollnutzungsstunden**")
                    col3, col4 = st.columns(2)
                    col3.metric("Leistungspreis", f"{demand_charge_low:,.2f} €/kW")
                    col4.metric("Arbeitspreis", f"{energy_charge_high_input:,.2f} ct/kW")

                    st.write(f"**\\>={VOLLLASTSTUNDEN_THRESHOLD} Vollnutzungsstunden**")
                    col3,col4 = st.columns(2)

                    col3.metric("Leistungspreis", f"{demand_charge_high:,.2f} €/kW")

                    col4.metric("Arbeitspreis", f"{energy_charge_low_input:,.2f} ct/kW")

                    st.subheader("☀️ PV Anlage")
                    if pv_total > 0:
                        st.metric("Leistung ", f"{pv_total:,.0f} kWp") 
                    elif custom_pv_file is not None:
                        st.metric("Leistung ", "Custom Profile")
                    else: 
                        st.write("Nicht konfiguriert")

        with tab_agg:

            fig_overview_gesamt = px.line(df, x="timestamp", y="load",
                                   title=f"Lastprofil ab {df['timestamp'].dt.year.min()}",
                                   labels={"timestamp": "Zeit", "load": "Last (kW)"}
                                         )
            fig_overview_gesamt.update_layout(height=400, xaxis_title="Zeit")
            st.plotly_chart(fig_overview_gesamt, use_container_width=True, config=plotly_config)


        with tab_solar:
            if pv_total == 0 and custom_pv_file is None:
                st.warning("Keine PV konfiguriert (bitte in der Seitenleiste linke angegeben).")
            else:
                fig_pv = go.Figure()

                # Add PV load  in yellow
                fig_pv.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df_pv["yearly_production_kw"],
                    mode='lines',
                    name='PV Erzeugung',
                    line=dict(color='yellow')
                ))

                fig_pv.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['load'],
                    mode='lines',
                    name='Lastgang',
                    line=dict(color='black')
                ))

                # Add grid load  in green
                fig_pv.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['load_pv'],
                    mode='lines',
                    name='Lastgang mit PV',
                    line=dict(color='green')
                ))

                # Add Einspeisung
                fig_pv.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df["load_pv_neg"],
                    mode='lines',
                    name='Eingespeiste Energie',
                    line=dict(color='orange')
                ))

                # Add Netzbezug
                fig_pv.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df["load_pv_pos"],
                    mode='lines',
                    name='Netzbezug',
                    line=dict(color='blue')
                ))


                # Optional layout tweaks
                fig_pv.update_layout(
                    title='Lastgang mit und ohne PV',
                    xaxis_title='Zeit',
                    yaxis_title='Last (kW)',
                    template='simple_white'
                )

                with st.container(border=True):
                    col1_pv, col2_pv, col3_pv, col4_pv, col5_pv, col6_pv = st.columns(6)

                    with col1_pv:
                        st.header("Ohne PV ")
                        st.header("Mit PV ", help="Nur Netzbezug")
                    with col2_pv:
                        st.metric("Energieverbrauch", f"{total_energy_Mwh:,.0f} MWh")
                        st.metric("Netto Energieverbrauch", f"{positive_load_pv.sum() / 4 / 1000:,.0f} MWh", f"{-(total_energy_Mwh-(positive_load_pv.sum() / 4 / 1000)):,.0f} MWh", delta_color="inverse")
                    with col3_pv:
                        st.metric("Spitzenlast", f"{df['load'].max():,.0f} kW")
                        st.metric("Spitzenlast", f"{positive_load_pv.max():,.0f} kW", f"{-(peak_load-positive_load_pv.max()):,.0f} kW", delta_color="inverse")
                    with col4_pv:
                        st.metric("Volllaststunden", f"{(volllaststunden):,.0f} h")
                        st.metric("Volllaststunden", f"{volllaststunden_pv:,.0f} h")
                    with col5_pv:
                        st.metric(f" ","")
                        st.metric(f" ","")
                        st.metric("Bezug aus PV", f"{(total_energy_Mwh - positive_load_pv.sum() / 4 / 1000):,.0f} MWh")
                    with col6_pv:
                        st.metric(f" ","")
                        st.metric(f" ","")
                        st.metric("Netzeinspeisung", f"{-(negative_load_pv.sum() / 4 / 1000):,.0f} MWh")

                with st.container(border=True):
                    col1_pv, col2_pv, col3_pv, col4_pv, col5_pv, col6_pv = st.columns(6)

                    with col1_pv:
                        st.header("PV Daten")
                    with col2_pv:
                        if pv_total > 0:
                            st.metric("Nennleistung", f"{pv_total:,.0f} kWp")
                        else:
                            st.metric("Max. Leistung", f"{df_pv['yearly_production_kw'].max():,.0f} kWp")
                    with col3_pv:
                        st.metric("Erzeugte Energie", f"~{df_pv['yearly_production_kw'].sum() / 4 / 1000:,.0f} MWh")
                    with col4_pv:
                        st.metric("Eigenverbrauchsquote PV", f"{pv_self_consumption_ratio:,.0f} %")
                    with col5_pv:
                        st.metric("Autarkiegrad", f"{(df['load'].sum() - (positive_load_pv.sum()))/(df['load'].sum() )*100:,.0f} %", help="Anteil des Gesamtverbrauchs, der durch PV gedeckt wurde")

                st.plotly_chart(fig_pv, use_container_width=True, config=plotly_config)


        with tab_toppeaks:
            st.subheader("🚩  Spitzenlasten im Lastgangprofil")
            df_peaks = df.copy()

            col1, col2, col3 = st.columns([2, 3, 6], vertical_alignment="top")
            with col1:
                with st.container(border = True):
                    # Let user select number of entries shown
                    n_peaks = st.number_input("Anzahl der Spitzen", min_value=1, value=30)

                    top_peaks = df_peaks.nlargest(n_peaks, "load").reset_index(drop=True)[["timestamp", "Date", "time",  "Weekday", "load"]]
                    st.metric("Höchster Wert", f"{top_peaks['load'].max():,.0f} kW")
                    st.metric("Niedrigster Wert", f"{top_peaks['load'].min():,.0f} kW")

            with col2:
                st.write(top_peaks[["Date", "time", "Weekday","load"]])

            with col3:
                year = top_peaks["timestamp"].dt.year.min()
                # Define start and end of the year
                start = datetime.datetime(year, 1, 1)
                end = datetime.datetime(year, 12, 31, 23, 00, 00)

                fig_top20 = px.bar(top_peaks.sort_values("load"), x="timestamp", y="load",
                                   title=f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten",
                                   labels={"load": "Last (kW)", "timestamp": "Zeit"})
                fig_top20.update_layout(xaxis_tickformat="%b",xaxis_title=f"{year}", xaxis_tickangle=-45, xaxis=dict(nticks=20, range=[start, end]))
                st.plotly_chart(fig_top20, use_container_width=True)


            ################### +++++++++++++++++++++++++++++SUN+++++++++++++++++++++++++++++ ###################
            if pv_total > 0 or custom_pv_file is not None:

                st.subheader("☀️ Spitzenlasten mit PV")
                df_pv_peaks = df.copy()
                col1, col2, col3 = st.columns([2, 3, 6], vertical_alignment="top")
                with col1:
                    with st.container(border = True):
                        # Let user select number of entries shown
                        #n_peaks = st.number_input("Anzahl der Spitzen", min_value=1, value=30)

                        top_peaks_pv = df_pv_peaks.nlargest(n_peaks, "load_pv").reset_index()[["timestamp", "Date", "time", "Weekday", "load_pv"]]
                        st.metric("Höchster Wert", f"{top_peaks_pv['load_pv'].max():,.0f} kW")
                        st.metric("Niedrigster Wert", f"{top_peaks_pv['load_pv'].min():,.0f} kW")

                with col2:
                    st.write(top_peaks_pv[["Date","time","Weekday","load_pv"]])

                with col3:
                    fig_top20_pv = px.bar(top_peaks_pv.sort_values("load_pv"), x="timestamp", y="load_pv",
                                       title=f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten",
                                       labels={"load_pv": "Last (kW)", "Timestamp": "Time"})
                    fig_top20_pv.update_layout(xaxis_tickformat="%b",xaxis_title=f"{year}", xaxis_tickangle=-45, xaxis=dict(nticks=20, range=[start, end]))
                    st.plotly_chart(fig_top20_pv, use_container_width=True)


                st.header("")
                col1, col3, col_leer = st.columns([3 , 8,1], vertical_alignment="top")
                with col1:
                    st.subheader(f"🚩☀️ {n_peaks} Spitzenlasten im Vergleich mit/ohne PV")
                    with st.container(border = True):
                        col_a, col_b = st.columns([2, 2])
                        #col_aa.subheader(" ")
                        #col_aa.write(" ")
                        #col_aa.subheader("Höchster Wert")
                        #col_aa.write(" ")
                        #col_aa.write(" ")
                        #col_aa.subheader("Niedrigster Wert")
                        col_a.subheader("Ohne PV")
                        #col_a.write("---")
                        col_a.metric("Spitzenlast ohne PV", f"{df['load'].max():,.0f} kW")
                        col_a.metric("Niedrigster Wert", f"{top_peaks['load'].min():,.0f} kW")

                        col_b.subheader("Mit PV")
                        #col_b.write("---")
                        col_b.metric("Spitzenlast mit PV", f"{top_peaks_pv['load_pv'].max():,.0f} kW")
                        col_b.metric("Niedrigster Wert", f"{top_peaks_pv['load_pv'].min():,.0f} kW")

                with col3:
                    st.subheader(f"📊 Übersicht der {n_peaks} höchsten Spitzenlasten unter Berücksichtigung einer PV-Anlage")

                    # Assume top_peaks and top_peaks_pv as before
                    timestamps_pv = set(top_peaks_pv['timestamp'])

                    # Assign colors and labels
                    top_peaks['color'] = top_peaks['timestamp'].apply(
                        lambda ts: 'red' if ts in timestamps_pv else 'green'
                    )
                    top_peaks['label'] = top_peaks['timestamp'].apply(
                        lambda ts: 'Nicht vermiedene Spitzenlasten' if ts in timestamps_pv else 'Vermiedene Spitzenlasten'
                    )

                    # Color map for German labels
                    color_discrete_map = {
                        'Nicht vermiedene Spitzenlasten': 'red',
                        'Vermiedene Spitzenlasten': 'green'
                    }

                    fig_combined = px.bar(
                        top_peaks.sort_values("load"),
                        x="timestamp",
                        y="load",
                        color="label",  # Use the German label column for legend
                        color_discrete_map=color_discrete_map,
                        title=f"",
                        labels={"load": "Power (kW)", "timestamp": "Zeit", "label": "Legende"}
                    )

                    fig_combined.update_layout(
                        xaxis_tickformat="%b",
                        xaxis_title="Zeitpunkt",
                        xaxis_tickangle=-45,
                        xaxis=dict(nticks=20, range=[start, end])
                    )

                    st.plotly_chart(fig_combined, use_container_width=True)


            st.header("🔍 Detail-Analyse der Spitzenlasten")

            col_context_left, col_context_right, col_context_buffer = st.columns([1, 3, 5])
            with col_context_left:
#                st.subheader(f"\n")
                st.subheader("Spitzenlast\n\n")
                selected_timestamp = st.selectbox("Bitte timestamp auswählen", top_peaks["timestamp"].astype(str))
                # Convert back to datetime if needed
                selected_timestamp = pd.to_datetime(selected_timestamp)
                # Sort full df by timestamp
                df_sorted = df_peaks.copy()
                df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

                # Find index of selected timestamp in full sorted DataFrame
                selected_index = df_sorted[df_sorted["timestamp"] == selected_timestamp].index

                if not selected_index.empty:
                    idx = selected_index[0]
                    # Get 10 rows before and after
                    context_df = df_sorted.iloc[max(0, idx - 10): idx + 11]

                    with col_context_right:
                        st.subheader(f"📊 10 Lasten vor und nach Spitzenlast \n\n **(am {selected_timestamp})**")
                       # st.dataframe(context_df[["date","Weekday","load"]])
                    #st.dataframe(context_df)
                        def highlight_selected_row(row):
                            if row["timestamp"] == selected_timestamp:
                                return ["background-color: #fdd835"] * len(row)  # yellow highlight
                            else:
                                return [""] * len(row)

                        # Apply the style
                        styled_context_df = context_df[["timestamp","Weekday","load"]]
                        styled_context_df = styled_context_df.style.apply(highlight_selected_row, axis=1)

                        # Show it in Streamlit
                        st.dataframe(styled_context_df)
                else:
                    st.warning("Selected timestamp not found in full data.")

            with col_context_buffer:
                st.subheader("Lastkurve\n\n")
                selected_day = selected_timestamp.date()
                df_day = df_peaks[df_peaks["timestamp"].dt.date == selected_day]
                fig_day = px.line(df_day, x="timestamp", y="load", title=f"🔋 Lastkurve am ausgewählten Tag ({selected_day})")
                fig_day.update_layout(height=500, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_day, use_container_width=True)

        with tab_loadduration:
            st.subheader("📉 Lastdauerkurve")
            sorted_loads = df_peaks["load"].sort_values(ascending=False).reset_index(drop=True)
            fig_load_duration = px.line(sorted_loads, title="🔺 Lastdauerkurve (ohne PV)")
            fig_load_duration.update_layout(yaxis_title="Last (kW)", xaxis_title="Anzahl Stunden (sortiert nach Last)")
            st.plotly_chart(fig_load_duration, use_container_width=True)
            st.write("\n \n")

            if pv_total > 0 or custom_pv_file is not None:
                st.subheader("📉☀️  Lastdauerkurve mit PV ️️☀️")
                sorted_loads_pv = df_peaks["load_pv"].sort_values(ascending=False).reset_index(drop=True)
                fig_load_duration_pv = px.line(sorted_loads_pv, title="🔺 Lastdauerkurve (mit PV)")
                fig_load_duration_pv.update_layout(yaxis_title="Last (kW)", xaxis_title="Anzahl Stunden (sortiert nach Last)")
                st.plotly_chart(fig_load_duration_pv, use_container_width=True)
                st.write("\n \n")

                st.subheader("Lastdauerkurve mit und ohne PV ")
                df_duration = pd.DataFrame({
                    "Ohne PV": sorted_loads,
                    "Mit PV": sorted_loads_pv
                })

                # Create figure
                fig = go.Figure()

                # Add Ohne PV (black)
                fig.add_trace(go.Scatter(
                    y=df_duration["Ohne PV"],
                    x=df_duration.index,
                    mode='lines',
                    name='Ohne PV',
                    line=dict(color='black')
                ))

                # Add Mit PV (orange)
                fig.add_trace(go.Scatter(
                    y=df_duration["Mit PV"],
                    x=df_duration.index,
                    mode='lines',
                    name='Mit PV',
                    line=dict(color='orange')
                ))

                fig.update_layout(
                    title="🔺 Lastdauerkurve mit und ohne PV",
                    yaxis_title="Last (kW)",
                    xaxis_title="Stunden (sortiert nach Last)",
                    legend_title="Legende"
                )

                st.plotly_chart(fig, use_container_width=True)
            ##########################################



        # Add additional information about the load profile
        with st.expander("Additional load profile statistics"):
            st.write(f"**Data Points:** {len(df)}")
            st.write(f"**Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            st.write(f"**Median Load:** {df['load'].median():.2f} kW")
            st.write(f"**Standard Deviation:** {df['load'].std():.2f} kW")
            st.write(f"**Load Factor:** {(avg_load / peak_load * 100):.2f}%")


#################################################################################################################################################################



with taboptimierer:

    if uploaded_file:
        st.header("🔋 Batterie-Optimierer")
        with st.expander("Info", expanded=False):
            st.info("Hier können Sie einsehen, wie hoch die maximal mögliche Spitzenreduktion mit einer bestimmten Batterie ist.")
        
        # Battery selection UI with immediate results
        col_left, col_right = st.columns([1, 5])
        
        with col_left:
            st.subheader("Batterie-Auswahl")
            
            # Initialize session state for battery selection
            if 'selected_battery' not in st.session_state:
                st.session_state.selected_battery = None
            
            # Vertical battery selection buttons
            button_1_clicked = st.button("🔋 \n\n 100 kW | 215 kWh", type="primary" if st.session_state.selected_battery == 1 else "secondary")
            button_2_clicked = st.button("🔋🔋 \n\n 200 kW | 430 kWh", type="primary" if st.session_state.selected_battery == 2 else "secondary")
            button_3_clicked = st.button("🔋🔋🔋 \n\n 300 kW | 645 kWh", type="primary" if st.session_state.selected_battery == 3 else "secondary")
            button_4_clicked = st.button("🔋🔋🔋🔋 \n\n 400 kW | 860 kWh", type="primary" if st.session_state.selected_battery == 4 else "secondary")
            button_5_clicked = st.button("🔋🔋🔋🔋🔋 \n\n 500 kW | 1075 kWh", type="primary" if st.session_state.selected_battery == 5 else "secondary")
            button_6_clicked = st.button("💚 \n\n 600 kW | 1290 kWh", type="primary" if st.session_state.selected_battery == 6 else "secondary")
            button_7_clicked = st.button("💚 \n\n 700 kW | 1505 kWh", type="primary" if st.session_state.selected_battery == 7 else "secondary")
            button_9_clicked = st.button("💚 \n\n 900 kW | 1935 kWh", type="primary" if st.session_state.selected_battery == 9 else "secondary")
            button_8_clicked = st.button("💚 \n\n 800 kW | 1720 kWh", type="primary" if st.session_state.selected_battery == 8 else "secondary")
            button_10_clicked = st.button("💚 \n\n 1000 kW | 2150 kWh", type="primary" if st.session_state.selected_battery == 10 else "secondary")
            button_11_clicked = st.button("💚 \n\n 1100 kW | 2365 kWh", type="primary" if st.session_state.selected_battery == 11 else "secondary")
            button_12_clicked = st.button("💚 \n\n 1200 kW | 2580 kWh", type="primary" if st.session_state.selected_battery == 12 else "secondary")
            button_13_clicked = st.button("💚 \n\n 1300 kW | 2795 kWh", type="primary" if st.session_state.selected_battery == 13 else "secondary")
            button_14_clicked = st.button("💚 \n\n 1400 kW | 3010 kWh", type="primary" if st.session_state.selected_battery == 14 else "secondary")
            button_15_clicked = st.button("💚 \n\n 1500 kW | 3225 kWh", type="primary" if st.session_state.selected_battery == 15 else "secondary")


            # Handle button clicks
            if button_1_clicked:
                st.session_state.selected_battery = 1
                st.rerun()
            elif button_2_clicked:
                st.session_state.selected_battery = 2
                st.rerun()
            elif button_3_clicked:
                st.session_state.selected_battery = 3
                st.rerun()
            elif button_4_clicked:
                st.session_state.selected_battery = 4
                st.rerun()
            elif button_5_clicked:
                st.session_state.selected_battery = 5
                st.rerun()
            elif button_6_clicked:
                st.session_state.selected_battery = 6
                st.rerun()
            elif button_7_clicked:
                st.session_state.selected_battery = 7
                st.rerun()
            elif button_8_clicked:
                st.session_state.selected_battery = 8
                st.rerun()
            elif button_9_clicked:
                st.session_state.selected_battery = 9
                st.rerun()
            elif button_10_clicked:
                st.session_state.selected_battery = 10
                st.rerun()
            elif button_11_clicked:
                st.session_state.selected_battery = 11
                st.rerun()
            elif button_12_clicked:
                st.session_state.selected_battery = 12
                st.rerun()
            elif button_13_clicked:
                st.session_state.selected_battery = 13
                st.rerun()
            elif button_14_clicked:
                st.session_state.selected_battery = 14
                st.rerun()
            elif button_15_clicked:
                st.session_state.selected_battery = 15
                st.rerun()
        
        with col_right:
            # Show analysis results immediately when battery is selected
            if st.session_state.selected_battery:
                battery_specs = {
                    1: {"power": 100, "capacity": 215},
                    2: {"power": 200, "capacity": 430},
                    3: {"power": 300, "capacity": 645},
                    4: {"power": 400, "capacity": 860},
                    5: {"power": 500, "capacity": 1075},
                    6: {"power": 600, "capacity": 1290},
                    7: {"power": 700, "capacity": 1505},
                    8: {"power": 800, "capacity": 1720},
                    9: {"power": 900, "capacity": 1935},
                    10: {"power": 1000, "capacity": 2150},
                    11: {"power": 1100, "capacity": 2365},
                    12: {"power": 1200, "capacity": 2580},
                    13: {"power": 1300, "capacity": 2795},
                    14: {"power": 1400, "capacity": 3010},
                    15: {"power": 1500, "capacity": 3225}
                }
                
                selected_specs = battery_specs[st.session_state.selected_battery]
                
                # Run optimization directly
                with st.spinner("Optimierung läuft..."):
                    # Set battery parameters
                    opt_battery_power_kw = selected_specs['power']
                    opt_battery_capacity_kwh = selected_specs['capacity']
                    
                    # Prepare numpy array for faster processing
                    load_array = df["load"].values  # Convert to numpy array
                    peak_load_org = load_array.max()
                    peak_load = peak_load_org  # Match original variable name from your attached selection
                    total_consumption_optimizer = load_array.sum() / 4
                    
                    # EXACT COPY of battery_simulation_ps but using numpy arrays for speed
                    def battery_simulation_ps_numpy(load_data, battery_capacity, power_rating, threshold_kw, depth_of_discharge=DEFAULT_DEPTH_OF_DISCHARGE, battery_efficiency=BATTERY_EFFICIENCY):
                        # EXACT ORIGINAL LOGIC
                        total_capacity = battery_capacity  # kWh
                        reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 10%) in kWh
                        soc = total_capacity  # start fully charged in kWh
                        interval_hours = INTERVAL_HOURS
                        
                        threshold_kw = load_data.max() - threshold_kw
                        
                        optimized = []
                        charge = []
                        discharge = []
                        soc_state = []
                        
                        for load in load_data:
                            grid_load = load  # start with original load
                            
                            # --- DISCHARGING --- (EXACT ORIGINAL)
                            if load > threshold_kw and soc > reserve_energy:
                                power_needed = load - threshold_kw
                                max_discharge_power = (soc - reserve_energy) / interval_hours
                                actual_discharge_power = min(power_rating, power_needed, max_discharge_power)
                                
                                energy_used = actual_discharge_power * interval_hours / battery_efficiency
                                # energy_used = actual_discharge_power * interval_hours / 1
                                soc = soc - energy_used
                                
                                grid_load = load - actual_discharge_power
                                charge.append(0)
                                discharge.append(actual_discharge_power)
                                
                            # --- CHARGING (only when load is below threshold to avoid peak increase) --- (EXACT ORIGINAL)
                            elif load <= threshold_kw and soc < total_capacity:
                                
                                max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold
                                
                                max_charge_power = (total_capacity - soc) / interval_hours
                                actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)
                                
                                energy_stored = actual_charge_power * interval_hours * battery_efficiency
                                soc = min(soc + energy_stored, total_capacity)
                                
                                grid_load = load + actual_charge_power
                                charge.append(actual_charge_power)
                                discharge.append(0)
                                
                            else:
                                charge.append(0)
                                discharge.append(0)
                            
                            optimized.append(grid_load)
                            soc_state.append(soc)
                        
                        return np.array(optimized)
                    
                    # === ORIGINAL PEAK SHAVING OPTIMIZATION ===
                    # Test different peak reduction values - EXACT ORIGINAL
                    all_results = []
                    min_reduction_ps = min(0.2 * opt_battery_power_kw, 10)
                    max_reduction_ps = min(opt_battery_power_kw, peak_load)
                    # FIX: Include the endpoint by adding 1 to max_reduction_ps
                    reduction_values_ps = np.arange(min_reduction_ps, max_reduction_ps + 1, 1)
                    
                    best_peak_reduction = -np.inf
                    best_threshold_ps_only = 0
                    
                    for ps_reduction_value in reduction_values_ps:
                        # Run numpy-based battery simulation
                        optimized_load = battery_simulation_ps_numpy(
                            load_array, opt_battery_capacity_kwh, opt_battery_power_kw, 
                            threshold_kw=ps_reduction_value, depth_of_discharge=DEFAULT_DEPTH_OF_DISCHARGE, battery_efficiency=BATTERY_EFFICIENCY
                        )
                        
                        # Calculate results after peak shaving
                        peak_after_ps = optimized_load.max()
                        peak_reduction_ps = peak_load - peak_after_ps
                        
                        # EXACT FROM YOUR ATTACHED SELECTION
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
                    
                    # Find best threshold
                    if all_results:
                        best = max(all_results, key=lambda x: x["kW_reduction"])
                        best_threshold_ps_only = best["threshold_kW"]
                    
                    best_threshold_kw = best_threshold_ps_only
                    
                    # Create df_opt for visualization (only needed for chart)
                    df_opt = df.copy()
                    df_opt["net_load_kw"] = df_opt["load"]
                
                # Fancy results display
                st.success(f"✅ Optimierung für Batterie mit {selected_specs['power']} kW und {selected_specs['capacity']} kWh abgeschlossen!")
                
                # Create fancy KPI cards
                st.markdown("### ⭐ Optimierungsergebnisse")
                
                # Main KPI container with subtle styling
                with st.container():
                    col2, col1, col3 = st.columns(3)
                    
                    with col1:
                        # Peak reduction - main KPI
                        reduction_percentage = (best_peak_reduction / peak_load) * 100
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #1f77b4; font-size: 1.5rem; margin-bottom: 0.3rem;">⚡</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Lastspitzenreduktion</div>
                            <div style="color: #333; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.2rem;">{best_peak_reduction:.1f} kW</div>
                            <div style="color: #888; font-size: 0.8rem;">({reduction_percentage:.1f}% Reduktion)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Original peak
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #ff6b6b; font-size: 1.5rem; margin-bottom: 0.3rem;">📈</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Ursprüngliche Lastspitze</div>
                            <div style="color: #333; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.2rem;">{peak_load:.1f} kW</div>
                            <div style="color: #888; font-size: 0.8rem;">Vor Optimierung</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Optimized peak
                        optimized_peak = peak_load - best_peak_reduction
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #4CAF50; font-size: 1.5rem; margin-bottom: 0.3rem;">🎯</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Optimierte Lastspitze</div>
                            <div style="color: #333; font-size: 1.8rem; font-weight: bold; margin-bottom: 0.2rem;">{optimized_peak:.1f} kW</div>
                            <div style="color: #888; font-size: 0.8rem;">Nach Optimierung</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Enhanced visualization
                #st.subheader("Lastgang mit optimalem Threshold")
                
                # Create enhanced load profile chart with timestamps - FULL RESOLUTION
                fig = go.Figure()
                
                # Create timestamp index for full year (assuming 15-minute intervals starting from Monday)
                import pandas as pd
                from datetime import datetime, timedelta
                
                # Create a realistic timestamp range for full year with full resolution
                start_time = datetime(2024, 1, 1, 0, 0)  # Start on a Monday
                timestamps = [start_time + timedelta(minutes=15*i) for i in range(len(df_opt))]
                
                # Original load with full resolution to show all peaks and threshold violations
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=df_opt["load"],
                    mode='lines',
                    name='Lastgang (vollständig)',
                    line=dict(color='#2E86AB', width=1),
                    hovertemplate='<b>%{x|%A, %d.%m.%Y %H:%M}</b><br>Last: %{y:.1f} kW<extra></extra>'
                ))
                
                # Optimal threshold line
                fig.add_hline(
                    y=best_threshold_kw,
                    line_dash="dash",
                    line_color="#E63946",
                    line_width=2,
                    annotation_text=f"Optimaler Threshold: {best_threshold_kw:.1f} kW",
                    annotation_position="top right"
                )
                
                # Layout for yearly view with full resolution
                fig.update_layout(
                    title={
                        'text': f'Jahres-Lastgang mit {st.session_state.selected_battery} Fox G-MAX Batterie-Optimierung',
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': '#2c3e50'}
                    },
                    xaxis_title='Zeitraum (2024)',
                    yaxis_title='Leistung (kW)',
                    template='plotly_white',
                    height=450,
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        tickformat='%b %Y',  # Show month and year for yearly view
                        dtick="M1",  # Show every month
                        tickangle=45,
                        type='date'
                    ),
                    yaxis=dict(
                        gridcolor='#f0f0f0',
                        gridwidth=1
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    # Performance optimizations
                    uirevision='constant',  # Prevents unnecessary re-renders
                    dragmode='pan'  # Optimize for panning instead of zoom by default
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Technical Parameters - moved below graph with box styling
                st.markdown("#### 🔧 Technische Parameter")
                
                # Create styled technical parameter cards
                with st.container():
                    col_tech1, col_tech2, col_tech3 = st.columns(3)
                    
                    with col_tech1:
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #4CAF50; font-size: 1.5rem; margin-bottom: 0.3rem;">🔋</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Batterie-Konfiguration</div>
                            <div style="color: #333; font-size: 1.4rem; font-weight: bold; margin-bottom: 0.2rem;">{selected_specs['capacity']} kWh</div>
                            <div style="color: #888; font-size: 0.8rem;">{selected_specs['power']} kW / C-Rate: 0,5 </div>
                        </div>
                        """, unsafe_allow_html=True)

                        #<div style="color: #333; font-size: 1.4rem; font-weight: bold; margin-bottom: 0.2rem;">{st.session_state.selected_battery} Fox G-MAX</div>
                    
                    with col_tech2:
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #FF9800; font-size: 1.5rem; margin-bottom: 0.3rem;">⚙️</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Optimale Spitzenlast</div>
                            <div style="color: #333; font-size: 1.4rem; font-weight: bold; margin-bottom: 0.2rem;">{best_threshold_kw:.1f} kW</div>
                            <div style="color: #888; font-size: 0.8rem;">Schwellenwert für Spitzenkappung</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_tech3:
                        efficiency_score = min(100, (best_peak_reduction / selected_specs['power']) * 100)
                        st.markdown(f"""
                        <div style="
                            background: #fafafa;
                            border: 2px solid #e0e0e0;
                            padding: 1rem;
                            border-radius: 8px;
                            text-align: center;
                            margin-bottom: 0.5rem;
                        ">
                            <div style="color: #2196F3; font-size: 1.5rem; margin-bottom: 0.3rem;">📊</div>
                            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.2rem;">Effizienz-Score</div>
                            <div style="color: #333; font-size: 1.4rem; font-weight: bold; margin-bottom: 0.2rem;">{efficiency_score:.1f}%</div>
                            <div style="color: #888; font-size: 0.8rem;">Batterie-Ausnutzung (nur Spitzenkappung)</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("👈 Wählen Sie eine Batterie-Konfiguration aus")





with tabsimulation:
    # ----------------------------------- PARAMETER EINGABE LINKS ----------------------------------------
    if uploaded_file:
        # Create test for battery
        #df_bt_org = df.copy()
        df_exp_2 = df.copy()
        df_exp_pv = df_with_pv.copy()

        st.header("🔋 📉 Batterie-Simulation")
       # st.warning("**Disclaimer**: This dashboard is intended for **indicative purposes only**. "
       #            "All calculations and results are based on **simplified assumptions** and **should not be interpreted as precise or reliable forecasts.** "
       #            "The tool provides no guarantee of accuracy, performance, or financial outcomes. Users must validate all assumptions independently and conduct a **thorough technical and financial analysis** before making any decisions based on the results shown. "
       #            "The creators of this dashboard accept no liability for any actions taken based on its outputs.")

        # ----------- Battery configurations -----------------------
        battery_lifetime = 10

        if 'battery_capacity' not in st.session_state:
            st.session_state.battery_capacity = 215

        if 'power_rating' not in st.session_state:
            st.session_state.power_rating = 100

        if 'battery_cost_per_kwh' not in st.session_state:
            st.session_state.battery_cost_per_kwh = 210

        battery_capacity = 0
        power_rating = 0
        battery_cost_per_kwh = 210

        # Functions to update battery values
        def set_small():
            st.session_state.battery_capacity = 90
            st.session_state.power_rating = 92
            st.session_state.battery_cost_per_kwh = 400

        def set_medium():
            st.session_state.battery_capacity = 215
            st.session_state.power_rating = 100
            st.session_state.battery_cost_per_kwh = 210

        def set_large():
            st.session_state.battery_capacity = 430
            st.session_state.power_rating = 200
            st.session_state.battery_cost_per_kwh = 210

        def set_xlarge():
            st.session_state.battery_capacity = 645
            st.session_state.power_rating = 300
            st.session_state.battery_cost_per_kwh = 210


        def set_custom():
            # This just activates the custom input section
            st.session_state.show_custom = True if st.session_state.show_custom == False else st.session_state.show_custom == True


        col_settings, col_simulation = st.columns([1, 5])

        with col_settings:

            with st.container(border=True):

                st.markdown("### 🔧🔋 Batterie")
                with st.expander("Auswahl"):
                    # Create simple buttons stacked vertically
                    st.button("Voltfang (90 kWh / 92 kW)", on_click=set_small)
                    st.button("1 Fox G-MAX (215 kWh / 100 kW)", on_click=set_medium)
                    st.button("2 Fox G-MAX (430 kWh / 200 kW)", on_click=set_large)
                    st.button("3 Fox G-MAX (645 kWh / 300 kW)", on_click=set_xlarge)

                    if 'show_custom' not in st.session_state:
                        st.session_state.show_custom = False
                    st.button("Eigene Konfiguration", on_click=set_custom)

                    if st.session_state.show_custom:
                        st.session_state.battery_capacity = st.number_input("Kapazität (kWh)", min_value=1, value=st.session_state.battery_capacity)

                        st.session_state.power_rating = st.number_input(
                            "Leistung (kW)",
                            min_value=1,
                            value=st.session_state.power_rating
                        )

                    # Display the current values
                st.write(f"Gewählte Kapazität: **{st.session_state.battery_capacity} kWh**")
                st.write(f"Gewählte Leistung: **{st.session_state.power_rating} kW**")
                st.write(f"Entladungstiefe: **{DEFAULT_DEPTH_OF_DISCHARGE}%**")
                st.write(f"Roundtrip-Effizienz: **{int(BATTERY_EFFICIENCY * 100)}%**")
                with st.expander("**Kostenannahmen**"):
                    st.session_state.battery_cost_per_kwh = st.number_input("Batterie Kosten pro kWh (€/kWh)", min_value=100, value=st.session_state.battery_cost_per_kwh)
                    system_cost_multiplier = st.number_input("Systemkosten Zusatzfaktor", min_value=1.0, value=1.2)

            ## --------------------------------- PEAK REDUCTION --------------------------------------------------------------------
            with st.container(border=True):
                st.subheader("🔻 Spitzenreduktion")
                st.write(f"Max. Reduktion mit gewählter Batterie: **{st.session_state.power_rating}kW**")

                value_peak_reduction = st.number_input("Reduktion um (kW) ",
                                                       0,
                                                       int(peak_load) * 1000000,
                                                       st.session_state.power_rating if st.session_state.power_rating < int(0.3 * peak_load) else int(0.3 * peak_load))

                calculated_peakshaving_threshold = (peak_load - value_peak_reduction) / peak_load *100
                if pv_total > 0 or custom_pv_file is not None:
#                    st.write(f"Ziel-Spitzenlast ohne PV: {(peak_load - value_peak_reduction):.0f}kW")
                    st.write(f"Spitzenlast inkl. PV: {(peak_load_pv):,.0f}kW")
                st.write(f"Anteil Spitzenlast : {100-calculated_peakshaving_threshold:,.0f}%")

            ## -------------------------------------------  FINANCIAL ASSUMPTIONS ----------------------------------------------------------------------------


            with st.container(border=True):
                st.markdown("### 💰 Netzentgelte", help="Bitte in Seitenleiste links eingeben")
                with st.expander("Details"):
                    st.write(f"**\\<= {VOLLLASTSTUNDEN_THRESHOLD} VLH**")
                    st.write(f"Arbeitspreis: {energy_charge_high} ct/kWh, Leistungspreis: {demand_charge_low} €/kw")
                    st.write(f"**> {VOLLLASTSTUNDEN_THRESHOLD} VLH**")
                    st.write(f"Arbeitspreis: {energy_charge_low} ct/kWh, Leistungspreis > {VOLLLASTSTUNDEN_THRESHOLD} VLH: {demand_charge_high} €/kw")

            ## -------------------------------------------  MODIFICATON MONTHS ----------------------------------------------------------------------------
            with st.container(border=True):
                st.markdown("### 🌻 PV Priorisierung ")
                start_month_pv = st.number_input("Start (Monat)", value=0, min_value=0, max_value=13)
                end_month_pv = st.number_input("Ende (Monat)", value=0, min_value=0, max_value=13)


            ###############################################################################################################################

        # ----------------------------------- MAIN WINDOW RECHTS  ----------------------------------------
        with col_simulation:
            if (st.session_state.battery_capacity == 0 and st.session_state.power_rating == 0):
                with st.container(border=True):
                    st.subheader("Bitte Batteriegröße auswählen")
            # -----------------------------        PEAKSHAVING METRICS CALCULATION   -----------------------------------------------------

            df_peakshaving = battery_simulation_v02(df.copy(), st.session_state.battery_capacity, st.session_state.power_rating, DEFAULT_DEPTH_OF_DISCHARGE, calculated_peakshaving_threshold)


            # --------------------------------   METRICS CALCULATION------------------------------------------------
            # Peaks
            peak_org = peak_load
            peak_peakshaving = df_peakshaving["grid_load"].max()
            peak_reduction_peakshaving = peak_org - peak_peakshaving

            # Consumption
            grid_energy_peakshaving = df_peakshaving["load"].sum() / 4
            volllaststunden_peakshaving = grid_energy_peakshaving / peak_peakshaving

            # Finance
            demand_charge_peakshaving = demand_charge_high if volllaststunden_peakshaving > VOLLLASTSTUNDEN_THRESHOLD else demand_charge_low
            energy_charge_peakshaving = energy_charge_low if volllaststunden_peakshaving > VOLLLASTSTUNDEN_THRESHOLD else energy_charge_high
            annual_savings_actual = peak_reduction_peakshaving * demand_charge_peakshaving
            
            total_battery_cost = st.session_state.battery_capacity * st.session_state.battery_cost_per_kwh * system_cost_multiplier
            payback_period = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
            payback_period_target = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
            roi = (annual_savings_actual * battery_lifetime - total_battery_cost) / total_battery_cost * 100

            # --------------------------------   TABS ------------------------------------------------
            b_without_pv,  b_with_pv_ebo, b_atypik = st.tabs(["Ohne PV - Nur Spitzenlastkappung", "Mit PV - Detailansicht", "Atypische Netznutzung"])

            with b_without_pv:
                #if (st.session_state.battery_capacity == 0 and st.session_state.power_rating == 0):
                #    with st.container(border=True):
                #        st.subheader("Bitte Batteriegröße auswählen")
                #else:

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(
                        "<div class='metric'><div class='metric-title'>🔺 Alte Spitzenlast</div><div class='metric-value'>" + f"{peak_org:,.1f} kW" + "</div></div>",
                        unsafe_allow_html=True)

                with col2:
                    st.markdown(
                        "<div class='metric'><div class='metric-title'>🔻 Neue Spitzenlast</div><div class='metric-value'>" + f"{peak_peakshaving:,.1f} kW" + "</div></div>",
                        unsafe_allow_html=True)

                with col3:
                    st.markdown(
                        "<div class='metric'><div class='metric-title'>📉 Reduktion</div><div class='metric-value'>" +
                        f"{(peak_reduction_peakshaving):.0f}kW   ({-(peak_reduction_peakshaving / peak_org * 100):.1f}%)" + "</div></div>",
                        unsafe_allow_html=True)
                with col4:
                    st.markdown(
                        "<div class='metric' style='background-color:#e6f9ec; border-radius:8px; padding:8px; border:2px solid #4caf50;'><div class='metric-title'>💸 Ersparnis (p.a.)</div><div class='metric-value'>" + f"€{demand_charge_peakshaving * peak_reduction_peakshaving:,.1f}" + "</div></div>",
                        unsafe_allow_html=True,
                        help="Kalkuliert durch Spitzenlastkappung mit der Annahme einer Reduktion von " + f"{peak_reduction_peakshaving:,.1f} kW" + " und einem Leistungspreis von  " + f"€/kW{demand_charge_peakshaving:,.1f}" + " pro kW pro Jahr.")

                st.write("---")

                # ---------------------------------------    BATTERIE CHART OHNE PV           ---------------------------------------
                if (peak_reduction_peakshaving) == 0 :
                    st.warning("⚠️ Keine Lastspitzenreduktion erfolg. Die gewählte Batteriespezifikation ist zu gering für Ihr Lastprofil. \n "
                               "Wählen Sie eine Batterie mit höherer Kapazität oder passen Sie den Grenzwert zur Spitzenlastkappung an.⚠️")
                elif (peak_reduction_peakshaving) < 0.98 * value_peak_reduction:
                    st.warning(f"Spitzenlast reduziert um: {peak_reduction_peakshaving:,.0f}kW"
                                "⚠️ Die ausgewählte Batteriegröße reicht nicht aus, um die Spitze um den gewünschten Wert zu senken. \n "
                               "Wählen Sie eine Batterie mit mehr Kapazität und Leistung oder reduzieren Sie den Grenzwert zur Spitzenlastkappung.⚠️")

                legend_rename = {
                    "load": "Last (kW)",
                    "grid_load": "Netto Netzlast",
                    # "battery_soc": "Batterie Ladestand",
                    "battery_discharge": "Batterie Entladung"
                }
                df_peakshaving_renamed = df_peakshaving.rename(columns=legend_rename)

                color_map = {
                    "Last (kW)": "#eb1b17",
                    "Netto Netzlast": "#11a64c",
                    "Batterie Entladung": "#030ca8",
                    "Batterie Ladestand": "#e64e02"
                }


                def add_ref_line(fig, y, text, color):
                    fig.add_trace(
                        go.Scatter(
                            x=[df_peakshaving_renamed["timestamp"].min(), df_peakshaving_renamed["timestamp"].max()],
                            y=[y, y],
                            mode="lines",
                            name=text,
                            line=dict(dash="dot", color=color),
                        )
                    )

                fig_battery_peakshaving = go.Figure()

                # Add traces
                for col in legend_rename.values():
                    fig_battery_peakshaving.add_trace(
                        go.Scattergl(
                            x=df_peakshaving_renamed["timestamp"],
                            y=df_peakshaving_renamed[col],
                            mode="lines",
                            name=col,
                            line=dict(shape="linear", color=color_map.get(col))
                        )
                    )


                # Helper to add horizontal reference lines
                def add_ref_line_old(fig, y, text, color, xref="timestamp"):
                    fig.add_shape(
                        type="line",
                        x0=df_peakshaving_renamed["timestamp"].min(), x1=df_peakshaving_renamed["timestamp"].max(),
                        y0=y, y1=y,
                        line=dict(dash="dot", color=color)
                    )
                    fig.add_annotation(
                        x=df_peakshaving_renamed["timestamp"].max(), y=y,
                        text=text, showarrow=False,
                        xanchor="right", yanchor="bottom",
                        font=dict(color=color)
                    )


                # Conditional + fixed reference lines
                if peak_peakshaving > (peak_org - value_peak_reduction)*1.01 :
                    add_ref_line(fig_battery_peakshaving, peak_peakshaving,
                                 f"Spitzenlast mit Batterie: ({peak_peakshaving:.0f}kW)", "orange")

                add_ref_line(fig_battery_peakshaving, calculated_peakshaving_threshold * 0.01 * peak_org,
                             f"Ziel Spitzenlast ({peak_org - value_peak_reduction:.1f}kW)",
                             "red")

                add_ref_line(fig_battery_peakshaving, peak_org,
                             f"Ursprüngliche Spitzenlast ({peak_org:.0f}kW)", "grey")

                # Layout
                fig_battery_peakshaving.update_layout(
                    title="⚡ Analyse des Lastprofils mit Spitzenlastkappung",
                    yaxis_title="Leistung (kW)",
                    xaxis_title="Zeit",
                    height=500,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(title="Legend")
                )

                st.plotly_chart(fig_battery_peakshaving, use_container_width=True)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ZUSAMMENFASSUNG BATTERIE OHNE PV CASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                with st.container(border=True):
                    st.subheader("📈 Zusammenfassung")
                    col0, col1, col2, col3, col4 = st.columns(5)
                    with col0:
                        st.subheader("Verbrauch")
                        st.metric("⚡ Gesamtverbrauch", f"{(df_peakshaving['load'].sum() / 1000 / 4):,.0f} MWh")

                    with col1:
                        st.subheader("Spitzenlast")
                        st.metric("🔺 Spitzenlast ohne Batterie", f"{df['load'].max():,.0f} kW")
                        st.metric("🔸 Spitzenlast mit Batterie", f"{df_peakshaving['grid_load'].max():,.0f} kW",
                                  f"{df_peakshaving['grid_load'].max() - df['load'].max():,.0f} kW", delta_color="inverse")
                    with col2:
                        col2.subheader("Volllaststunden")
                        st.metric("Ohne Batterie", f"{df_org['load'].sum() / 4 / peak_org:,.0f} h")
                        st.metric("Mit Batterie", f"{df_org['load'].sum() / 4 / peak_peakshaving:,.0f} h")
                    with col3:
                        st.subheader("Finanzen")
                        st.metric(f"💰 Jährl. Einsparungen (Peak Shaving)",
                                  f"~€ {annual_savings_actual:,.0f}", help = f"(Leistungspreis: {demand_charge}€/kW)")
                        st.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")
                    with col4:
                        st.subheader("ROI")
                        st.metric("✅ Amortisationszeit", f"{payback_period:.1f} Jahre")
                        st.metric(f"📈 ROI über {battery_lifetime} Jahre (Batterie Lebenszeit)", f"{roi:.1f}%")


                with st.container(border=True):
                    st.subheader("📈 Finanzielle Analyse")
                    col2, col3, col4 = st.columns(3)

                    #col1.subheader("Batteriekosten")
                    #col1.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")

                    col2.subheader("Ersparnis")
                    col2.metric(f"💰 Geschätzte jährl. Einsparungen durch Peak Shaving",
                                  f"~€ {annual_savings_actual:,.1f}", help = f"(Leistungspreis: {demand_charge}€/kW)")

                    col3.subheader("Amortisationszeit")
                    col3.metric("✅ Amortisationszeit", f"{payback_period:.1f} Jahre")

                    col4.subheader("ROI")
                    col4.metric(f"📈 ROI über {battery_lifetime} Jahre (Batterie Lebenszeit)", f"{roi:.1f}%")

                    st.write("---")
                    col_1c, col_2c, col_3c = st.columns(3)
                    col_1c.subheader("Batteriekosten")
                    col_1c.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")
                    col_2c.subheader("Energiekosten")
                    col_2c.metric("Arbeitspreis", f"{energy_charge_low if volllaststunden_peakshaving > VOLLLASTSTUNDEN_THRESHOLD else energy_charge_high:.2f} €")
                    col_3c.subheader(" ")
                    col_3c.metric("Leistungspreis", f"{demand_charge_high if volllaststunden_peakshaving > VOLLLASTSTUNDEN_THRESHOLD else demand_charge_low:.1f} €")

                with st.container(border=True):
                    st.subheader("🔋 Batterie Analyse")
                    col_b_1, col_b_2, col_b_3 = st.columns(3)

                    col_b_1.subheader("🔋 Geladene Energie")
                    charge_energy = df_peakshaving["battery_charge"].sum()
                    col_b_1.metric("Aus Netz", f"{charge_energy:,.1f}")

                    col_b_2.subheader("🪫 Entladene Energie")
                    discharge_energy = df_peakshaving["battery_discharge"].sum()
                    col_b_2.metric("Für Peakshaving", f"{discharge_energy:,.1f}")

        #################################################################################################################################################################
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ BATTERIE SIMULATION PV CASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            with b_with_pv_ebo:

               # df_exp_2 = battery_simulation_vpv(df_with_pv, st.session_state.battery_capacity,
               #                                   st.session_state.power_rating, 90,
               #                                   calculated_peakshaving_threshold)

                calculated_peakshaving_threshold_pv = ((peak_load_pv - value_peak_reduction) / peak_load_pv * 100) if (pv_total > 0 or custom_pv_file is not None) else ((peak_org - value_peak_reduction) / peak_org * 100)

                df_exp_2 = battery_simulation_vpv_selfconsumption_working(df_with_pv, st.session_state.battery_capacity,
                                                                          st.session_state.power_rating, DEFAULT_DEPTH_OF_DISCHARGE,
                                                                          calculated_peakshaving_threshold_pv)

               ########################        VARIABLEN FÜR BATTERIE SIMILATION           ###########################

                volllaststunden_bt_pv_2 = df_exp_2["grid_load_pv_bt"].clip(lower=0).sum() / 4 / df_exp_2["grid_load_pv_bt"].max()

                demand_charge = demand_charge_high if volllaststunden_bt_pv_2 > VOLLLASTSTUNDEN_THRESHOLD else demand_charge_low
                energy_charge = energy_charge_low if volllaststunden_bt_pv_2 > VOLLLASTSTUNDEN_THRESHOLD else energy_charge_high

                energy_exported_mwh = -df_exp_2["grid_load_pv_bt"].clip(upper=0).sum() / 4 / 1000
                pv_selfcons_kwh = min(
                    df_exp_2['battery_charge_pv_selfcons'].sum() / 4,
                    df_exp_2['battery_discharge_selfcons'].sum() / 4
                )
                annual_savings_selfcons = pv_selfcons_kwh * energy_charge
                annual_savings_peakshaving = (peak_load_pv - df_exp_2["grid_load_pv_bt"].max()) * demand_charge



                annual_savings_total = annual_savings_peakshaving + annual_savings_selfcons
                payback_period = total_battery_cost / annual_savings_total
                roi = (annual_savings_total * battery_lifetime - total_battery_cost) / total_battery_cost * 100

                max_load_jump = df["load"].diff().clip(lower=0).max()


            ########################################         KPI Übersicht über Graph           ####################################################
                col_0, col_1, col_2, col_3, col_4 = st.columns([3,3,4,4,3])
                with col_0:
                    with st.container(border=True):
                        st.subheader("⚡ Netzbezug")
                        #st.metric("Netzbezug mit PV", f"{energy_consumed_pv:,.0f} MWh")
                        st.metric("Netzbezug mit PV & Batterie",
                                  f"{df_exp_2['grid_load_pv_bt'].clip(lower=0).sum() / 4 / 1000:,.0f} MWh",
                                  f"{df_exp_2['grid_load_pv_bt'].clip(lower=0).sum() / 4 / 1000 - energy_consumed_pv:,.0f} MWh",delta_color="inverse")
                        st.write(f"(Netzbezug ohne Batterie: **{energy_consumed_pv:,.0f}** MWh)\n\n")

                with col_1:
                    with st.container(border=True):
                        st.subheader("🔺 Spitzenlast")
                        #st.metric("Spitzenlast mit PV", f"{peak_load_pv:,.0f} kW")
                        st.metric("Erreichte Spitzenlast mit PV & Batterie", f"{df_exp_2['grid_load_pv_bt'].max():,.0f} kW",
                                  f"{df_exp_2['grid_load_pv_bt'].max()-peak_load_pv:,.0f} kW",delta_color="inverse")
                        st.write(f"(Spitzenlast ohne Batterie: **{peak_load_pv:,.0f}** kW)\n\n")

                with col_2:
                    with st.container(border=True):
                        vollstring2 = f">= {VOLLLASTSTUNDEN_THRESHOLD} h" if volllaststunden_bt_pv_2 > VOLLLASTSTUNDEN_THRESHOLD else f"< {VOLLLASTSTUNDEN_THRESHOLD} h"
                        demand_charge2 = demand_charge_high if volllaststunden_bt_pv_2 > VOLLLASTSTUNDEN_THRESHOLD else demand_charge_low

                        st.subheader("⏺️ Volllaststunden")
                        #st.metric("Mit PV", f"{vollaststunden_pv:,.0f} h")
                        st.metric("Mit PV & Batterie", f"{volllaststunden_bt_pv_2:,.0f} h", f"{(volllaststunden_bt_pv_2 - volllaststunden_pv):,.0f} h", delta_color="off", help=f"Resuliert in Leistungspreis von {demand_charge2:,.0f} €/kW ")
                        st.write(f"(VLH ohne Batterie: **{volllaststunden_pv:,.0f}** h)\n\n")

                with col_3:
                    with st.container(border=True):
                        st.subheader("💰 Einsparpotential")
                        savings_ps = (peak_load_pv - df_exp_2["grid_load_pv_bt"].max()) * demand_charge2
                        st.metric("Järhliche Einsparungen", f"~{(savings_ps + annual_savings_selfcons):,.0f}€")
                                #f"~{savings_ps:,.0f} €"
                                #f" Durch Reduktion von {(peak_load_pv - df_exp_2["grid_load_pv_bt"].max()):,.0f}kW",
                                #f"~{savings_ps:,.0f} €")
                        st.write(f" **{savings_ps:,.0f}€** : Lastreduktion um {(peak_load_pv - df_exp_2['grid_load_pv_bt'].max()):,.0f}kW \n\n  **{annual_savings_selfcons:,.0f}€** : Eigenbedarfsoptimierung" )

                with col_4:
                    with st.container(border=True):
                        st.subheader("✅ ROI")
                        st.metric("Amortisationszeit", f"{payback_period:.1f} Jahre")


                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OPTIMIERUNG GRAPH PV CASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                

                st.write("---")
                df_exp_2_optimized = df_exp_2.copy()
                grid_load_max = df_exp_2["grid_load_pv_bt"].max()
                load_org_max = df_exp_2["load_org"].max()
                load_max = df_exp_2["load"].max()
                target_peak = calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv

                columns_to_plot = ["load", "grid_load_pv_bt", "load_pv_neg", "battery_discharge", "battery_charge"]

                column_labels = [
                "Netzlast (mit PV) ohne Batterie",
                "Netzlast mit Batterie",
                "PV-Überschuss",
                "Batterie-Entladung",
                "Batterie-Ladung (gesamt)"
                ]
                df_plot = df_exp_2_optimized[columns_to_plot].copy()
                df_plot.columns = column_labels
                fig_batt_exp_pv_2 = go.Figure()
                colors = {
                    "Netzlast (mit PV) ohne Batterie": "crimson",
                    "Netzlast mit Batterie": "#43A047",
                    "Batterie-Ladung (gesamt)": "#64B5F6",
                    "Batterie-Entladung": "#1976D2",
                    "PV-Überschuss": "gold"
                }

                # Add traces more efficiently
                for col in column_labels:
                    fig_batt_exp_pv_2.add_trace(
                        go.Scattergl(
                            x=df_plot.index,
                            y=df_plot[col],
                            mode='lines',
                            name = col,
                            line = dict(color=colors[col], shape='linear'),
                            connectgaps = True
                        )
                    )

                # Add horizontal lines more efficiently - group similar operations
                hlines_data = [
                    (load_org_max, "dot", "DarkSlateGray", "Spitzenlast ohne PV", "top right"),
                    (load_max, "dot", "red", "Spitzenlast mit PV", "top right"),
                    (target_peak, "dot", "MediumVioletRed", f"Ziel-Spitzenlast ({target_peak:.0f}kW)", "bottom left"),
                (grid_load_max, "dot", "chartreuse", f"Erreichte Spitzenlast ({grid_load_max:.0f}kW)", "bottom right")
                ]

                for y_val, dash, color, text, position in hlines_data:
                    fig_batt_exp_pv_2.add_hline(
                        y=y_val,
                        line_dash=dash,
                        line_color=color,
                        annotation_text=text,
                        annotation_position=position,
                        name=text,
                        showlegend=True
                    )

                    # Update layout once at the end
                fig_batt_exp_pv_2.update_layout(
                    title="⚡ Netzlast, Batterieeinsatz und Spitzenlasten mit PV & Batterie ☀️",
                    xaxis_title = "Zeit",
                    yaxis_title = "Leistung (kW)",
                    hovermode = 'x unified',
                    showlegend = True
                )

                # Display the optimized figure
                st.plotly_chart(fig_batt_exp_pv_2, use_container_width=True)


                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DETAIL GRAPHEN PV ANSICHT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


                with st.expander("Detail-Ansicht"):
                    fig_detailed_pv_bt = go.Figure()

                    # Netzlast ohne PV und Batterie
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load_org'],
                        mode='lines', name=' Netzlast ohne PV und Batterie',
                        line=dict(color='darkgray', width=1),
                        hovertemplate=' Netzlast ohne PV und Batterie: %{y:.1f} kW'
                    ))

                    # Netzlast mit PV ohne Batterie (dunkelrot)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load'].clip(lower=0),
                        mode='lines', name='Netzlast mit PV',
                        line=dict(color='darkred', width=1),
                        hovertemplate='Netzlast (mit PV): %{y:.1f} kW'
                    ))

                    # Netzeinspeisung mit PV ohne Batterie (rot)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['load'].clip(upper=0),
                        mode='lines', name='Netzeinspeisung mit PV',
                        line=dict(color='crimson', width=1),
                        hovertemplate='Netzeinspeisung (mit PV): %{y:.1f} kW'
                    ))

                    # Netzlast mit Batterie (grün)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['grid_load_pv_bt'].clip(lower=0),
                        mode='lines', name='Netzlast (mit PV & Batterie)',
                        line=dict(color='darkgreen', width=2),
                        hovertemplate='Netzlast mit Batterie: %{y:.1f} kW'
                    ))
                    # Netzeinspeisung mit Batterie (grün)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['grid_load_pv_bt'].clip(upper=0),
                        mode='lines', name='Netzeinspeisung (mit PV & Batterie)',
                        line=dict(color='springgreen', width=2),
                        hovertemplate='Netzeinspeisung mit Batterie: %{y:.1f} kW'
                    ))

                    # Batterie-Ladung mit PV (gelb)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_charge_pv'],
                        mode='lines', name='Batterie-Ladung aus PV',
                        line=dict(color='mediumblue', width=1),
                        hovertemplate='Batterie-Ladung aus PV: %{y:.1f} kW'
                    ))

                    # Batterie-Ladung von Grid )
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_charge_grid'],
                        mode='lines', name='Batterie-Ladung aus Netz',
                        line=dict(color='purple', width=1),
                        hovertemplate='Batterie-Ladung aus Netz: %{y:.1f} kW'
                    ))

                    # Batterie-Discharge
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_discharge'],
                        mode='lines', name='Batterie-Entladung',
                        line=dict(color='midnightblue', width=1),
                        hovertemplate='Batterie-Entladung: %{y:.1f} kW'
                    ))


                    # Batterie-Ladung (orange, gestrichelt, sekundäre Y-Achse)
                    fig_detailed_pv_bt.add_trace(go.Scatter(
                        x=df_exp_2['timestamp'], y=df_exp_2['battery_soc'],
                        mode='lines', name='Batterie-Ladung (SoC)',
                        line=dict(color='#ff780a', width=2),
                        yaxis='y2',
                        hovertemplate='SoC: %{y:.1f} kWh'
                    ))


                    # Horizontale Linien für Spitzenlasten
                    fig_detailed_pv_bt.add_hline(
                        y=df_exp_2["grid_load_pv_bt"].max(), line_dash="dot", line_color="green",
                        annotation_text=f"Erreichte Spitzenlast ({df_exp_2['grid_load_pv_bt'].max():.0f} kW)",
                        annotation_position="bottom right"
                    )

                    if (df_exp_2["grid_load_pv_bt"].max() != (calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv)):
                        fig_detailed_pv_bt.add_hline(
                            y=calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv, line_dash="dot", line_color="blue",
                            annotation_text=f"Ziel-Spitzenlast ({calculated_peakshaving_threshold_pv * 0.01 * peak_load_pv:.0f} kW)",
                            annotation_position="bottom left"
                        )

                    fig_detailed_pv_bt.add_hline(
                        y=df_exp_2["load"].max(), line_dash="dot", line_color="red",
                        annotation_text=f"Spitzenlast ohne Batterie ({df_exp_2['load'].max():.0f} kW)",
                        annotation_position="top right"
                    )

                    # Layout-Optimierung
                    fig_detailed_pv_bt.update_layout(
                        title="⚡☀️🔋🔎 Detail-Ansicht: Netzlast, Batterieeinsatz und Spitzenlasten mit PV & Batterie",
                        xaxis_title="Zeit",
                        yaxis_title="Leistung (kW)",
                        yaxis2=dict(
                            title="Batterie-Ladung (kWh)",
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        legend_title="Legende",
                        height=550,
                        margin=dict(l=20, r=20, t=60, b=20),
                        font=dict(size=14)
                    )

                    st.plotly_chart(fig_detailed_pv_bt, use_container_width=True)

                with st.expander("Monatliche Batterie-Nutzung"):
                    monthly_summary = df_exp_2.groupby('month').agg({
                        'battery_charge_pv': 'sum',
                        'battery_charge_grid': 'sum',
                        'battery_discharge_peakshave': 'sum',
                        'battery_discharge_selfcons': 'sum'
                    }) / 4

                    fig_month = go.Figure()
                    fig_month.add_trace(go.Bar(name='Laden aus PV', x=monthly_summary.index,
                                               y=monthly_summary['battery_charge_pv']))
                    fig_month.add_trace(go.Bar(name='Laden aus Netz', x=monthly_summary.index,
                                               y=monthly_summary['battery_charge_grid']))
                    fig_month.add_trace(go.Bar(name='Entladen für Peak Shaving', x=monthly_summary.index,
                                               y=monthly_summary['battery_discharge_peakshave']))
                    fig_month.add_trace(go.Bar(name='Entladen für Eigenverbrauch', x=monthly_summary.index,
                                               y=monthly_summary['battery_discharge_selfcons']))

                    fig_month.update_layout(
                        barmode='group',
                        title="Monatliche Batterie-Nutzung nach Quelle und Zweck",
                        xaxis_title="Monat", yaxis_title="Energie (kWh)",
                        height=400, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_month, use_container_width=True)

                with st.expander("Batterie Ladezyklen"):
                    import plotly.subplots as sp
                    fig_batt = sp.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                subplot_titles=("Batterie Laden", "Batterie Entladen"))

                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_charge_pv"],
                        mode='lines', name="Laden aus PV", line=dict(color="#f6d55c")
                    ), row=1, col=1)
                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_charge_grid"],
                        mode='lines', name="Laden aus Netz", line=dict(color="#3caea3")
                    ), row=1, col=1)

                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_discharge_peakshave"],
                        mode='lines', name="Entladen für Peak Shaving", line=dict(color="#ed553b")
                    ), row=2, col=1)
                    fig_batt.add_trace(go.Scatter(
                        x=df_exp_2["timestamp"], y=df_exp_2["battery_discharge_selfcons"],
                        mode='lines', name="Entladen für Eigenverbrauch", line=dict(color="#20639b")
                    ), row=2, col=1)

                    fig_batt.update_layout(
                        title="Batterie Lade- und Entladeleistung",
                        height=600, margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_batt, use_container_width=True)

               # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DETAIL BOXEN PV CASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                with st.container(border=True):
                    st.header("📈 Finanzanalyse")

                    col1, col2, col3, col4 = st.columns(4)

                    col1.subheader("Batteriekosten")
                    col1.metric("💶 Geschätzte Investmentkosten", f"~€ {total_battery_cost:,.0f}")

                    col2.subheader("Ersparnis")
                    col2.metric(
                        "💰 Peak Shaving",
                        f"~€ {annual_savings_peakshaving:,.0f}",
                        help=f"Einsparung durch Reduktion der Spitzenlast ({demand_charge} €/kW)"
                    )
                    col2.metric(
                        "💡 Eigenverbrauch",
                        f"~€ {annual_savings_selfcons:,.0f}",
                        help=f"Einsparung durch PV-Eigenverbrauch via Batterie ({energy_charge_high} €/kWh)"
                    )
                    col2.metric(
                        "💰 Gesamteinsparung",
                        f"~€ {annual_savings_total:,.0f}",
                        help="Summe aus Peak Shaving und Eigenverbrauch"
                    )

                    col3.subheader("ROI & Payback")
                    col3.metric("✅ Payback Zeitraum", f"{payback_period:.1f} years")
                    col3.metric("📈 ROI über Lebenszeit", f"{roi:.1f} %")

                    #                        col4.subheader("Parameter")
                    with col4.expander("Parameter", expanded=False):
                        st.metric("Strompreis", f"{energy_charge_high:.2f} €/kWh")
                        st.metric("Leistungspreis", f"{demand_charge:.2f} €/kW")
                        st.metric("Batterielebensdauer", f"{battery_lifetime} Jahre")

                with st.container(border=True):
                    st.header("📈 Lastgang Details")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.subheader("Energieverbrauch")
                        st.metric("Netzbezug ohne PV", f"{total_energy_kwh/1000:,.0f} MWh")
                        st.metric("Netzbezug mit PV", f"{df_exp_2['load_pv_pos'].sum()/4/1000:,.0f} MWh")
                        # // Change this to "drawn from battery instead
                        st.metric("Netzbezug mit PV & Batterie", f"{df_exp_2['grid_load_pv_bt'].clip(lower=0).sum()/4/1000:,.0f} MWh")

                    with col2:
                        st.subheader("Spitzenlast")
                        st.metric("🔺🔺Spitzenlast ohne PV", f"{peak_load:,.0f} kW")
                        st.metric("🔺 Spitzenlast mit PV", f"{peak_load_pv:,.0f} kW", f"{peak_load_pv - peak_load:,.0f} kW",
                                  delta_color="inverse")
                        st.metric("🔸 Spitzenlast mit PV & Batterie", f"{df_exp_2['grid_load_pv_bt'].max():,.0f} kW",
                                  f"{df_exp_2['grid_load_pv_bt'].max() - peak_load:,.0f} kW",
                                  delta_color="inverse")

                    with col3:
                        st.subheader("Eingespeiste Energie")
                        st.metric("Einspeisung ins Netz ohne Batterie", f"{-df_exp_2['load_pv_neg'].sum()/4/1000:,.0f} MWh",
                                 help="Summe aller Zeitpunkte mit negativer Netzlast mit PV (PV-Überschuss)")
                        st.metric("Einspeisung ins Netz nach Optimierung", f"{energy_exported_mwh:,.0f} MWh",
                                 help="Summe aller Zeitpunkte mit negativer Netzlast nach Batterie-Optimierung (PV-Überschuss nach Eigenverbrauch und Batterieladung)")
                    with col4:
                        st.subheader("Volllaststunden")
                        st.metric("Ohne Batterie & ohne PV", f"{df_org['load'].sum() / 4 / peak_org:,.0f} h")
                        st.metric("Mit PV",f"{df_exp_pv['load'].sum() / 4 / df_exp_pv['load'].max():,.0f} h" )
                        st.metric("Mit PV & Batterie", f"{df_exp_2['grid_load_pv_bt'].clip(lower=0).sum() / 4 / df_exp_2['grid_load_pv_bt'].max():,.0f} h")


                with st.container(border=True):
                    st.header("🔋 Batterie Analyse")
                    col_b_1, col_b_2, col_b_3, col_b_4 = st.columns(4)

                    with col_b_1:
                        st.subheader("🔋 Geladene Energie")
                        st.metric("Gesamt", f"{df_exp_2['battery_charge'].sum() / 4 /1000:,.0f} MWh")
                        st.metric("Aus Netz", f"{df_exp_2['battery_charge_grid'].sum() / 4 /1000:,.0f} MWh")
                        st.metric("Aus PV", f"{df_exp_2['battery_charge_pv'].sum() / 4 /1000:,.0f} MWh")


                    with col_b_2:
                        st.subheader("")
                        with st.expander("Details"):
                            st.metric("Aus PV (Selbstverbrauch)",
                                    f"{df_exp_2['battery_charge_pv_selfcons'].sum() / 4 /1000:,.0f} MWh")
                            st.metric("Aus PV (Peakshaving)",
                                    f"{df_exp_2['battery_charge_pv_peakshave'].sum() / 4 /1000:,.0f} MWh")
                            st.metric("Aus Netz (Selbstverbrauch)",
                                    f"{df_exp_2['battery_charge_grid_selfcons'].sum() / 4 /1000:,.0f} MWh")
                            st.metric("Aus Netz (Peakshaving)",
                                    f"{df_exp_2['battery_charge_grid_peakshave'].sum() / 4 /1000:,.0f} MWh")
                    with col_b_3:
                        st.subheader("🪫 Entladene Energie")
                        st.metric("Gesamt", f"{df_exp_2['battery_discharge'].sum() / 4 /1000:,.0f} MWh")
                    with col_b_4:
                        st.subheader("")
                        with st.expander("Details"):
                            st.metric("Für Peakshaving", f"{df_exp_2['battery_discharge_peakshave'].sum() / 4 /1000:,.0f} kWh")
                            st.metric("Für Eigenverbrauch", f"{df_exp_2['battery_discharge_selfcons'].sum() / 4 /1000:,.0f} kWh")

            with b_atypik:
                st.subheader("🕐 Atypische Netznutzung - Hochlastzeitfenster")
                st.write("Hier können Sie spezifische Zeitfenster definieren, in denen Lastspitzen für die Netzentgelte relevant sind.")
                
                # Preset buttons
                st.write("**Schnellauswahl:**")
                col_preset1, col_preset2, col_preset3, col_preset_spacer = st.columns([1, 1, 1, 3])
                
                with col_preset1:
                    if st.button("🔧 HLZ 2024", use_container_width=True):
                        # Winter: 7:15 - 12:00 and 16:30 - 19:15
                        st.session_state['enable_Winter'] = True
                        st.session_state['enable_Frühling'] = False
                        st.session_state['enable_Sommer'] = False
                        st.session_state['enable_Herbst'] = False
                        st.session_state['num_windows_Winter'] = 2
                        st.session_state['start_Winter_0'] = pd.Timestamp("07:15").time()
                        st.session_state['end_Winter_0'] = pd.Timestamp("12:00").time()
                        st.session_state['start_Winter_1'] = pd.Timestamp("16:30").time()
                        st.session_state['end_Winter_1'] = pd.Timestamp("19:15").time()
                        st.rerun()
                
                with col_preset2:
                    if st.button("🔧 HLZ 2025", use_container_width=True):
                        # Winter: 8:00 - 13:30 and 16:30 - 19:15
                        st.session_state['enable_Winter'] = True
                        st.session_state['enable_Frühling'] = False
                        st.session_state['enable_Sommer'] = False
                        st.session_state['enable_Herbst'] = False
                        st.session_state['num_windows_Winter'] = 2
                        st.session_state['start_Winter_0'] = pd.Timestamp("08:00").time()
                        st.session_state['end_Winter_0'] = pd.Timestamp("13:30").time()
                        st.session_state['start_Winter_1'] = pd.Timestamp("16:30").time()
                        st.session_state['end_Winter_1'] = pd.Timestamp("19:15").time()
                        st.rerun()
                
                with col_preset3:
                    if st.button("🔧 HLZ 2026", use_container_width=True):
                        # Winter: 7:15 - 12:00 and 16:30 - 19:15, Autumn: 17:30 - 18:00
                        st.session_state['enable_Winter'] = True
                        st.session_state['enable_Frühling'] = False
                        st.session_state['enable_Sommer'] = False
                        st.session_state['enable_Herbst'] = True
                        st.session_state['num_windows_Winter'] = 2
                        st.session_state['start_Winter_0'] = pd.Timestamp("07:15").time()
                        st.session_state['end_Winter_0'] = pd.Timestamp("12:00").time()
                        st.session_state['start_Winter_1'] = pd.Timestamp("16:30").time()
                        st.session_state['end_Winter_1'] = pd.Timestamp("19:15").time()
                        st.session_state['num_windows_Herbst'] = 1
                        st.session_state['start_Herbst_0'] = pd.Timestamp("17:30").time()
                        st.session_state['end_Herbst_0'] = pd.Timestamp("18:00").time()
                        st.rerun()
                
                st.write("---")
                
                # Time window configuration
                st.write("### Hochlastzeitfenster konfigurieren")
                
                # Define seasons with their months
                seasons = {
                    "Winter": [1, 2, 12],  # January, February, December
                    "Frühling": [3, 4, 5],  # March, April, May
                    "Sommer": [6, 7, 8],  # June, July, August
                    "Herbst": [9, 10, 11]  # September, October, November
                }
                
                # Collect all season configurations
                season_configs = {}
                
                # Create columns for all seasons in one row
                col_winter, col_fruehling, col_sommer, col_herbst = st.columns(4)
                season_columns = {
                    "Winter": col_winter,
                    "Frühling": col_fruehling,
                    "Sommer": col_sommer,
                    "Herbst": col_herbst
                }
                
                for season_name, months in seasons.items():
                    with season_columns[season_name]:
                        with st.container(border=True):
                            st.write(f"**{season_name}**")
                            
                            # Checkbox to enable/disable season
                            enabled = st.checkbox(
                                "Aktivieren",
                                value=(season_name == "Winter" or season_name == "Herbst"),  # Default: Winter and Autumn enabled
                                key=f"enable_{season_name}"
                            )
                            
                            if enabled:
                                # Define number of time windows for this season
                                num_windows = st.number_input(
                                    "Anzahl Fenster",
                                    min_value=1,
                                    max_value=5,
                                    value=2 if season_name == "Winter" else 1,  # Default: 2 for Winter, 1 for others
                                    key=f"num_windows_{season_name}"
                                )
                                
                                time_windows = []
                                for i in range(num_windows):
                                    st.write(f"*Fenster {i+1}:*")
                                    
                                    # Set default times based on season
                                    if season_name == "Winter":
                                        default_start = pd.Timestamp("07:15").time() if i == 0 else pd.Timestamp("16:45").time()
                                        default_end = pd.Timestamp("09:15").time() if i == 0 else pd.Timestamp("19:15").time()
                                    elif season_name == "Herbst":
                                        default_start = pd.Timestamp("17:30").time()
                                        default_end = pd.Timestamp("18:00").time()
                                    else:
                                        default_start = pd.Timestamp("08:00").time()
                                        default_end = pd.Timestamp("12:00").time()
                                    
                                    start_time = st.time_input(
                                        "Start",
                                        value=default_start,
                                        key=f"start_{season_name}_{i}",
                                        label_visibility="collapsed"
                                    )
                                    end_time = st.time_input(
                                        "Ende",
                                        value=default_end,
                                        key=f"end_{season_name}_{i}",
                                        label_visibility="collapsed"
                                    )
                                    time_windows.append((start_time, end_time))
                                
                                season_configs[season_name] = {
                                    "months": months,
                                    "time_windows": time_windows
                                }
                
                st.write("---")
                
                if season_configs:
                    # Create time window mask
                    df_atypik = df.copy()
                    df_atypik['month'] = df_atypik['timestamp'].dt.month
                    df_atypik['time'] = df_atypik['timestamp'].dt.time
                    
                    # Create mask for relevant time windows
                    window_mask = pd.Series(False, index=df_atypik.index)
                    
                    for season_name, config in season_configs.items():
                        for month in config['months']:
                            month_mask = df_atypik['month'] == month
                            for start_time, end_time in config['time_windows']:
                                if start_time <= end_time:
                                    time_mask = (df_atypik['time'] >= start_time) & (df_atypik['time'] <= end_time)
                                else:  # Handle overnight windows
                                    time_mask = (df_atypik['time'] >= start_time) | (df_atypik['time'] <= end_time)
                                window_mask |= (month_mask & time_mask)
                    
                    df_atypik['in_window'] = window_mask
                    
                    # Calculate peaks in and out of windows with date/time information
                    if df_atypik['in_window'].any():
                        peak_in_hlz_idx = df_atypik[df_atypik['in_window']]['load'].idxmax()
                        peak_in_hlz = df_atypik.loc[peak_in_hlz_idx, 'load']
                        peak_in_hlz_time = df_atypik.loc[peak_in_hlz_idx, 'timestamp']
                    else:
                        peak_in_hlz = 0
                        peak_in_hlz_time = None
                    
                    if (~df_atypik['in_window']).any():
                        peak_out_hlz_idx = df_atypik[~df_atypik['in_window']]['load'].idxmax()
                        peak_out_hlz = df_atypik.loc[peak_out_hlz_idx, 'load']
                        peak_out_hlz_time = df_atypik.loc[peak_out_hlz_idx, 'timestamp']
                    else:
                        peak_out_hlz = 0
                        peak_out_hlz_time = None
                    
                    st.write("---")
                    
                    # FIRST GRAPHIC: Load profile with HLZ windows and peak lines
                    st.write("### 📈 Lastprofil mit Hochlastzeitfenstern (HLZ)")
                    
                    fig_hlz_overview = go.Figure()
                    
                    # Add load profile
                    fig_hlz_overview.add_trace(
                        go.Scattergl(
                            x=df_atypik["timestamp"],
                            y=df_atypik["load"],
                            mode="lines",
                            name="Lastprofil",
                            line=dict(color="#eb1b17", width=1)
                        )
                    )
                    
                    # Add highlighted background for time windows
                    window_periods = []
                    current_start = None
                    
                    for i, (timestamp, in_window) in enumerate(zip(df_atypik['timestamp'], df_atypik['in_window'])):
                        if in_window and current_start is None:
                            current_start = timestamp
                        elif not in_window and current_start is not None:
                            window_periods.append((current_start, df_atypik['timestamp'].iloc[i-1]))
                            current_start = None
                    
                    # Handle case where window extends to end of data
                    if current_start is not None:
                        window_periods.append((current_start, df_atypik['timestamp'].iloc[-1]))
                    
                    # Add shaded regions for HLZ windows
                    for start, end in window_periods:
                        fig_hlz_overview.add_vrect(
                            x0=start, x1=end,
                            fillcolor="rgba(255, 255, 0, 0.3)",
                            layer="below",
                            line_width=0,
                        )
                    
                    # Add horizontal reference lines for peaks with date/time
                    if peak_in_hlz > 0 and peak_in_hlz_time is not None:
                        fig_hlz_overview.add_hline(
                            y=peak_in_hlz, 
                            line_dash="dot", 
                            line_color="red",
                            annotation_text=f"Peak in HLZ: {peak_in_hlz:.1f} kW<br>{peak_in_hlz_time.strftime('%d.%m.%Y %H:%M')}",
                            annotation_position="bottom right"
                        )
                    
                    if peak_out_hlz > 0 and peak_out_hlz_time is not None:
                        fig_hlz_overview.add_hline(
                            y=peak_out_hlz, 
                            line_dash="dash", 
                            line_color="blue",
                            annotation_text=f"Peak außerhalb HLZ: {peak_out_hlz:.1f} kW<br>{peak_out_hlz_time.strftime('%d.%m.%Y %H:%M')}",
                            annotation_position="top right"
                        )
                    
                    fig_hlz_overview.update_layout(
                        title="Lastprofil mit Hochlastzeitfenstern (gelb markiert)",
                        xaxis_title="Zeit",
                        yaxis_title="Leistung (kW)",
                        hovermode="x unified",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_hlz_overview, use_container_width=True)
                    
                    # Display enhanced metrics with date/time information
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        if peak_in_hlz > 0 and peak_in_hlz_time is not None:
                            st.metric("🔺 Peak in HLZ", f"{peak_in_hlz:,.1f} kW", 
                                     help=f"Aufgetreten am {peak_in_hlz_time.strftime('%d.%m.%Y um %H:%M Uhr')}")
                        else:
                            st.metric("🔺 Peak in HLZ", "0 kW")
                    
                    with col_info2:
                        if peak_out_hlz > 0 and peak_out_hlz_time is not None:
                            st.metric("🔺 Peak außerhalb HLZ", f"{peak_out_hlz:,.1f} kW",
                                     help=f"Aufgetreten am {peak_out_hlz_time.strftime('%d.%m.%Y um %H:%M Uhr')}")
                        else:
                            st.metric("🔺 Peak außerhalb HLZ", "0 kW")
                    
                    with col_info3:
                        # Calculate percentage difference between peaks
                        if peak_in_hlz > 0 and peak_out_hlz > 0:
                            peak_difference = ((peak_in_hlz - peak_out_hlz) / peak_out_hlz) * 100
                            if peak_difference > 0:
                                st.metric("📊 Peak-Differenz", f"+{peak_difference:.1f}%", 
                                         help=f"Peak in HLZ ist {peak_difference:.1f}% höher als außerhalb HLZ")
                            else:
                                st.metric("📊 Peak-Differenz", f"{peak_difference:.1f}%", 
                                         help=f"Peak in HLZ ist {abs(peak_difference):.1f}% niedriger als außerhalb HLZ")
                        elif peak_in_hlz > 0 and peak_out_hlz == 0:
                            st.metric("📊 Peak-Differenz", "∞", 
                                     help="Nur Peak in HLZ vorhanden")
                        elif peak_in_hlz == 0 and peak_out_hlz > 0:
                            st.metric("📊 Peak-Differenz", "-100%", 
                                     help="Kein Peak in HLZ vorhanden")
                        else:
                            st.metric("📊 Peak-Differenz", "0%", 
                                     help="Keine Peaks vorhanden")
                    
                    # Show detailed time window information
                    st.write("### 🕐 Definierte Hochlastzeitfenster")
                    
                    # Display configuration for each enabled season
                    for season_name, config in season_configs.items():
                        st.write(f"**{season_name}:**")
                        
                        months_names = {1: "Januar", 2: "Februar", 3: "März", 4: "April", 5: "Mai", 6: "Juni",
                                       7: "Juli", 8: "August", 9: "September", 10: "Oktober", 11: "November", 12: "Dezember"}
                        month_list = ", ".join([months_names[m] for m in config['months']])
                        
                        col_season1, col_season2 = st.columns(2)
                        with col_season1:
                            st.info(f"📅 {month_list}")
                        with col_season2:
                            for i, (start_time, end_time) in enumerate(config['time_windows']):
                                st.info(f"⏰ {start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}")
                    
                    # Show overall peak statistics
                    st.write("**📊 Peak-Übersicht:**")
                    
                    peak_stats = []
                    
                    # Add peak in HLZ
                    if peak_in_hlz > 0 and peak_in_hlz_time is not None:
                        peak_stats.append({
                            "Kategorie": "Peak in HLZ",
                            "Peak (kW)": f"{peak_in_hlz:.1f}",
                            "Zeitpunkt": peak_in_hlz_time.strftime('%d.%m.%Y %H:%M')
                        })
                    else:
                        peak_stats.append({
                            "Kategorie": "Peak in HLZ",
                            "Peak (kW)": "0",
                            "Zeitpunkt": "-"
                        })
                    
                    # Add peak outside HLZ
                    if peak_out_hlz > 0 and peak_out_hlz_time is not None:
                        peak_stats.append({
                            "Kategorie": "Peak außerhalb HLZ",
                            "Peak (kW)": f"{peak_out_hlz:.1f}",
                            "Zeitpunkt": peak_out_hlz_time.strftime('%d.%m.%Y %H:%M')
                        })
                    else:
                        peak_stats.append({
                            "Kategorie": "Peak außerhalb HLZ",
                            "Peak (kW)": "0",
                            "Zeitpunkt": "-"
                        })
                    
                    if peak_stats:
                        stats_df = pd.DataFrame(peak_stats)
                        st.dataframe(stats_df, use_container_width=True, hide_index=True)
                    
                    hlz_percentage = df_atypik['in_window'].sum() / len(df_atypik) * 100
                    st.info(f"🕐 **Gesamt:** {len(window_periods)} HLZ-Perioden mit {df_atypik['in_window'].sum()} Datenpunkten ({hlz_percentage:.1f}% der Zeit)")
                    
                    st.write("---")
                    
                    # BUTTON TO START BATTERY CALCULATION
                    if st.button("🔋 Batterie-Simulation für HLZ starten", type="primary", use_container_width=True):
                        
                        if len(df_atypik) > 0 and peak_in_hlz > 0:
                            st.write("### 🔋 Batterie-Simulation läuft...")
                            
                            
                            
                            # Calculate threshold based on HLZ peak minus target reduction
                            # Use the same absolute reduction as in the main calculation
                            target_reduction_kw = peak_org - (peak_org * calculated_peakshaving_threshold / 100)
                            target_peak_hlz = peak_in_hlz - target_reduction_kw
                            
                            # Ensure target is not negative
                            if target_peak_hlz < 0:
                                target_peak_hlz = peak_in_hlz * 0.8  # Fallback to 80% of HLZ peak
                            
                            # Calculate threshold percentage for HLZ peak
                            hlz_threshold_percentage = (target_peak_hlz / peak_in_hlz * 100) if peak_in_hlz > 0 else DEFAULT_DEPTH_OF_DISCHARGE
                            
                            # Run HLZ-only battery simulation
                            df_peakshaving_atypik = battery_simulation_hlz_only(
                                df_atypik.copy(), 
                                st.session_state.battery_capacity, 
                                st.session_state.power_rating, 
                                DEFAULT_DEPTH_OF_DISCHARGE, 
                                hlz_threshold_percentage
                            )
                            
                            # Calculate new peaks after battery simulation
                            peak_in_hlz_after = df_peakshaving_atypik[df_peakshaving_atypik['in_window']]['grid_load'].max() if df_peakshaving_atypik['in_window'].any() else 0
                            peak_out_hlz_after = df_peakshaving_atypik[~df_peakshaving_atypik['in_window']]['grid_load'].max() if (~df_peakshaving_atypik['in_window']).any() else 0
                            
                            # Calculate reductions
                            reduction_in_hlz = peak_in_hlz - peak_in_hlz_after
                            reduction_out_hlz = peak_out_hlz - peak_out_hlz_after
                            
                            # Display results
                            st.write("### 📊 Ergebnisse der Batterie-Simulation")
                            
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown(
                                    "<div class='metric'><div class='metric-title'>🔺 Peak in HLZ (vorher)</div><div class='metric-value'>" + f"{peak_in_hlz:,.1f} kW" + "</div></div>",
                                    unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(
                                    "<div class='metric'><div class='metric-title'>🔻 Peak in HLZ (nachher)</div><div class='metric-value'>" + f"{peak_in_hlz_after:,.1f} kW" + "</div></div>",
                                    unsafe_allow_html=True)
                            
                            with col3:
                                reduction_percentage = (reduction_in_hlz / peak_in_hlz * 100) if peak_in_hlz > 0 else 0
                                st.markdown(
                                    "<div class='metric'><div class='metric-title'>📉 Reduktion in HLZ</div><div class='metric-value'>" + 
                                    f"{reduction_in_hlz:.1f}kW ({reduction_percentage:.1f}%)" + "</div></div>",
                                    unsafe_allow_html=True)
                            
                            with col4:
                                annual_savings_hlz = demand_charge_peakshaving * reduction_in_hlz
                                st.markdown(
                                    "<div class='metric' style='background-color:#e6f9ec; border-radius:8px; padding:8px; border:2px solid #4caf50;'><div class='metric-title'>💸 Ersparnis (p.a.)</div><div class='metric-value'>" + f"€{annual_savings_hlz:,.1f}" + "</div></div>",
                                    unsafe_allow_html=True,
                                    help="Kalkuliert durch Spitzenlastkappung nur in den relevanten Hochlastzeitfenstern")
                            
                            # Summary table
                            st.write("### 📊 Detaillierte Übersicht")
                            summary_data = {
                                "Kategorie": ["In Hochlastzeitfenstern (HLZ)", "Außerhalb HLZ", "Gesamt"],
                                "Peak vorher (kW)": [f"{peak_in_hlz:,.1f}", f"{peak_out_hlz:,.1f}", f"{max(peak_in_hlz, peak_out_hlz):,.1f}"],
                                "Peak nachher (kW)": [f"{peak_in_hlz_after:,.1f}", f"{peak_out_hlz_after:,.1f}", f"{max(peak_in_hlz_after, peak_out_hlz_after):,.1f}"],
                                "Reduktion (kW)": [f"{reduction_in_hlz:,.1f}", f"{reduction_out_hlz:,.1f}", f"{reduction_in_hlz + reduction_out_hlz:,.1f}"],
                                "Reduktion (%)": [
                                    f"{reduction_percentage:.1f}%" if peak_in_hlz > 0 else "0%", 
                                    f"{(reduction_out_hlz / peak_out_hlz * 100):.1f}%" if peak_out_hlz > 0 else "0%",
                                    f"{((reduction_in_hlz + reduction_out_hlz) / max(peak_in_hlz, peak_out_hlz) * 100):.1f}%" if max(peak_in_hlz, peak_out_hlz) > 0 else "0%"
                                ],
                                "Relevanz für Netzentgelt": ["✅ Ja (HLZ)", "❌ Nein", "Gemischt"]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                            st.write("---")
                            
                            # Detailed visualization with battery simulation results
                            st.write("### 📈 Detaillierter Lastgang mit Batterie-Simulation")
                            
                            # Prepare data for plotting
                            df_plot = df_peakshaving_atypik.copy()
                            
                            legend_rename_atypik = {
                                "load": "Originallast (kW)",
                                "grid_load": "Netzlast mit Batterie (kW)",
                                "battery_discharge": "Batterie Entladung (kW)"
                            }
                            df_plot_renamed = df_plot.rename(columns=legend_rename_atypik)
                            
                            color_map_atypik = {
                                "Originallast (kW)": "#eb1b17",
                                "Netzlast mit Batterie (kW)": "#11a64c", 
                                "Batterie Entladung (kW)": "#030ca8"
                            }
                            
                            fig_atypik_detailed = go.Figure()
                            
                            # Add main traces
                            for col in legend_rename_atypik.values():
                                fig_atypik_detailed.add_trace(
                                    go.Scattergl(
                                        x=df_plot_renamed["timestamp"],
                                        y=df_plot_renamed[col],
                                        mode="lines",
                                        name=col,
                                        line=dict(shape="linear", color=color_map_atypik.get(col))
                                    )
                                )
                            
                            # Add shaded regions for HLZ windows
                            for start, end in window_periods:
                                fig_atypik_detailed.add_vrect(
                                    x0=start, x1=end,
                                    fillcolor="rgba(255, 255, 0, 0.2)",
                                    layer="below",
                                    line_width=0,
                                )
                            
                            # Add reference lines with date/time information
                            if peak_in_hlz > 0 and peak_in_hlz_time is not None:
                                fig_atypik_detailed.add_hline(
                                    y=peak_in_hlz, 
                                    line_dash="dot", 
                                    line_color="red",
                                    annotation_text=f"Peak in HLZ (vorher): {peak_in_hlz:.1f} kW<br>{peak_in_hlz_time.strftime('%d.%m.%Y %H:%M')}"
                                )
                            
                            fig_atypik_detailed.add_hline(
                                y=peak_in_hlz_after, 
                                line_dash="dot", 
                                line_color="green",
                                annotation_text=f"Peak in HLZ (nachher): {peak_in_hlz_after:.1f} kW"
                            )
                            
                            if peak_out_hlz > 0 and peak_out_hlz_time is not None:
                                fig_atypik_detailed.add_hline(
                                    y=peak_out_hlz, 
                                    line_dash="dash", 
                                    line_color="blue",
                                    annotation_text=f"Peak außerhalb HLZ: {peak_out_hlz:.1f} kW<br>{peak_out_hlz_time.strftime('%d.%m.%Y %H:%M')}"
                                )
                            
                            fig_atypik_detailed.update_layout(
                                title="Lastgang mit Batterie-Simulation und Hochlastzeitfenstern",
                                xaxis_title="Zeit",
                                yaxis_title="Leistung (kW)",
                                hovermode="x unified",
                                height=600,
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_atypik_detailed, use_container_width=True)
                            
                            # Battery usage metrics
                            st.write("### 🔋 Batterie-Nutzungsstatistiken")
                            col_b1, col_b2, col_b3, col_b4 = st.columns(4)
                            
                            with col_b1:
                                charge_energy = df_peakshaving_atypik["battery_charge"].sum() * INTERVAL_HOURS  # Convert to kWh
                                st.metric("🔌 Geladene Energie", f"{charge_energy:,.1f} kWh")
                            
                            with col_b2:
                                discharge_energy = df_peakshaving_atypik["battery_discharge"].sum() * INTERVAL_HOURS  # Convert to kWh
                                st.metric("🔋 Entladene Energie", f"{discharge_energy:,.1f} kWh")
                            
                            with col_b3:
                                discharge_in_hlz = df_peakshaving_atypik[df_peakshaving_atypik['in_window']]["battery_discharge"].sum() * INTERVAL_HOURS
                                st.metric("⚡ Entladung in HLZ", f"{discharge_in_hlz:,.1f} kWh")
                            
                            with col_b4:
                                efficiency = (discharge_energy / charge_energy * 100) if charge_energy > 0 else 0
                                st.metric("⚙️ Effizienz", f"{efficiency:.1f}%")
                        
                        else:
                            st.error("⚠️ Keine ausreichenden Daten für die Batterie-Simulation gefunden.")
                    
                    else:
                        st.info("👆 Klicken Sie auf den Button oben, um die Batterie-Simulation zu starten.")
                
                else:
                    st.info("👆 Bitte wählen Sie Monate und definieren Sie Zeitfenster, um die Analyse zu starten.")
                