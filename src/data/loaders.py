"""
Data loading functions.
"""

from typing import Union, Optional
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import INTERVAL_HOURS, CACHE_TTL

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading price data...")
def read_price_data(price_year: int = 2024) -> pd.DataFrame:
    """Read spot price data for specified year (2024 or 2025)
    
    Args:
        price_year: Year of price data to use (2024 or 2025)
    
    Returns:
        DataFrame with timestamp and price columns (price in EUR/kWh)
    
    Raises:
        ValueError: If price_year is not 2024 or 2025
    """
    if price_year == 2024:
        price_file = "data/spot_data_2024.xlsx"
        price_column = "Day-ahead Price (EUR/MWh)"
    elif price_year == 2025:
        price_file = "data/spot_data_2025.xlsx"
        price_column = "spotmarket"
    else:
        raise ValueError(f"Unsupported price year: {price_year}. Use 2024 or 2025.")
    
    df_prices = pd.read_excel(price_file)
    df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], format="mixed", dayfirst=True)
    
    # Remove timezone if present
    try:
        if df_prices['timestamp'].dt.tz is not None:
            df_prices['timestamp'] = df_prices['timestamp'].dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    
    # Convert price from EUR/MWh to EUR/kWh
    df_prices['price'] = pd.to_numeric(df_prices[price_column], errors='coerce') / 1000
    df_prices = df_prices[['timestamp', 'price']].dropna()
    
    return df_prices

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading load profile...")
def read_load_profile(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read load profile from Excel file or uploaded file object
    
    Args:
        file_path: Path to Excel file (string or Path) or file-like object
    
    Returns:
        DataFrame with columns: timestamp, load_org (kW), total_kwh
    
    Raises:
        ValueError: If file cannot be read or required columns not found
    """
    # Handle both file path (string) and uploaded file object
    df = pd.read_excel(file_path)
    
    # Try to auto-detect timestamp and load columns
    col_map = {col.lower(): col for col in df.columns}
    
    # Find timestamp column
    timestamp_col = None
    for key in ['timestamp', 'time', 'datum', 'date', 'timestamps', 'zeit']:
        if key in col_map:
            timestamp_col = col_map[key]
            break
    
    if timestamp_col is None:
        timestamp_col = df.columns[0]
    
    # Convert timestamp column with multiple fallback strategies
    try:
        df["timestamp"] = pd.to_datetime(df[timestamp_col], dayfirst=True)
    except Exception as e1:
        try:
            df["timestamp"] = pd.to_datetime(df[timestamp_col], utc=True)
        except Exception as e2:
            try:
                df["timestamp"] = pd.to_datetime(df[timestamp_col], format="mixed", dayfirst=True)
            except Exception as e3:
                try:
                    df["timestamp"] = pd.to_datetime(df[timestamp_col], format="%Y-%m-%dT%H:%M:%S%z")
                except Exception as e4:
                    try:
                        df["timestamp"] = pd.to_datetime(df[timestamp_col], format="ISO8601")
                    except Exception as e5:
                        raise ValueError(
                            f"Datei konnte nicht gelesen werden. Fehler:\n1. {e1}\n2. {e2}\n3. {e3}\n4. {e4}\n5. {e5}")
    
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
    
    # Try to find the correct column for load
    load_col = None
    for col in ['kWh', 'kwh', 'value', 'value_kwh', 'value_kw', 'load', 'kW', 'last', 'leistung', 'power']:
        if col in df.columns:
            load_col = col
            break
    
    if load_col is None:
        # Try case-insensitive search
        for key, col in col_map.items():
            if any(kw in key for kw in ['kw', 'load', 'value', 'last', 'leistung', 'power', 'wert']):
                load_col = col
                break
    
    if load_col is None:
        st.error("❌ Keine geeignete Last-Spalte gefunden. Bitte stellen Sie sicher, dass die Datei eine Spalte mit 'load', 'kW', 'kWh', oder 'value' enthält.")
        st.stop()
    
    # Convert load column to numeric
    df[load_col] = pd.to_numeric(df[load_col], errors='coerce').fillna(0)
    
    # Determine if the load column is in kW or kWh (15-min intervals)
    if load_col.lower() in ["load", "kw", "leistung", "power"]:
        df['load_org'] = df[load_col]
        df['total_kwh'] = df['load_org'] * INTERVAL_HOURS
    elif load_col.lower() in ["kwh", "value_kwh"]:
        # Assume kWh values for 15-minute intervals, convert to kW
        df['load_org'] = df[load_col] / INTERVAL_HOURS
        df['total_kwh'] = df['load_org'] * INTERVAL_HOURS
    else:
        # Default: assume it's kWh and convert to kW
        df['load_org'] = df[load_col] / INTERVAL_HOURS
        df['total_kwh'] = df['load_org'] * INTERVAL_HOURS
    
    return df

@st.cache_data(ttl=CACHE_TTL, show_spinner="Loading solar data...")
def load_solar_data(pv_total: float, custom_pv_file: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and process solar generation data
    
    Args:
        pv_total: Total PV capacity in kWp (used for standard profile scaling)
        custom_pv_file: Optional path to custom PV profile Excel file
    
    Returns:
        DataFrame with columns: timestamp, yearly_production_kw, yearly_production_kwh
    
    Raises:
        ValueError: If custom PV file is missing required columns
    """
    MAGIC_YEARLY_PV_MULTIPLIER = 850
    
    # Check if custom PV file is uploaded
    if custom_pv_file is not None:
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
            df_pv["timestamp"] = pd.to_datetime(df_pv[pv_timestamp_col], format="mixed", dayfirst=True)
        except Exception:
            try:
                df_pv["timestamp"] = pd.to_datetime(df_pv[pv_timestamp_col], utc=True)
            except Exception:
                df_pv["timestamp"] = pd.to_datetime(df_pv[pv_timestamp_col], dayfirst=True)
        
        # Remove timezone if present
        try:
            if df_pv["timestamp"].dt.tz is not None:
                df_pv["timestamp"] = df_pv["timestamp"].dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        
        df_pv["yearly_production_kw"] = pd.to_numeric(df_pv[pv_load_col], errors='coerce')
        df_pv["yearly_production_kwh"] = df_pv["yearly_production_kw"] * INTERVAL_HOURS
        
    else:
        # Use standard PV profile
        df_pv = pd.read_csv("data/solar_data_de_small.csv")
        df_pv["timestamp"] = pd.to_datetime(df_pv["timestamp"], format="%d.%m.%y %H:%M").dt.tz_localize(None)
        df_pv["yearly_production_kw"] = (
            df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) 
            * pv_total * MAGIC_YEARLY_PV_MULTIPLIER / INTERVAL_HOURS
        )
        df_pv["yearly_production_kwh"] = (
            df_pv["yearly_production_fraction"].astype(float).to_numpy().clip(min=0) 
            * pv_total * MAGIC_YEARLY_PV_MULTIPLIER
        )
    
    return df_pv[['timestamp', 'yearly_production_kw', 'yearly_production_kwh']]