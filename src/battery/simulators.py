"""
Shared battery simulation functions for peak shaving and load optimization.
"""
import pandas as pd

from src.config import INTERVAL_HOURS, BATTERY_EFFICIENCY


def battery_simulation_ps(
    df: pd.DataFrame,
    battery_capacity: float,
    power_rating: float,
    threshold_kw: float,
    depth_of_discharge: float,
    battery_efficiency: float = BATTERY_EFFICIENCY
) -> pd.DataFrame:
    """
    Peak shaving battery simulation function.
    
    Simulates battery charging and discharging to reduce peak loads below a threshold.
    Battery charges when load is below threshold (without exceeding it), and discharges
    when load exceeds threshold to bring it down.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with time-series load data. Must contain 'net_load_kw' column.
    battery_capacity : float
        Total battery capacity in kWh
    power_rating : float
        Maximum charge/discharge power in kW
    threshold_kw : float
        Target peak threshold in kW. Battery will try to cap loads at this level.
    depth_of_discharge : float
        Usable battery capacity as percentage (e.g., 90 means 90% DoD, 10% reserve)
    battery_efficiency : float
        Round-trip efficiency (default from config)
        
    Returns:
    --------
    pd.DataFrame
        Input dataframe with added columns:
        - ps_grid_load: Grid load after battery optimization
        - battery_charge: Charging power (kW)
        - battery_discharge: Discharging power (kW)
        - battery_soc: State of charge (kWh)
    """
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC
    soc = total_capacity  # start fully charged
    interval_hours = INTERVAL_HOURS

    # Calculate threshold from reduction target
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
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            charge.append(0)
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:
            max_possible_charge = threshold_kw - load  # Don't exceed threshold

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

