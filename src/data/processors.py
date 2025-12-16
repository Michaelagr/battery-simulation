"""
Data processing functions.
"""

import pandas as pd
import streamlit as st


def handle_german_dst_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Handle German DST transitions for load profile data.
    
    2024 German DST transitions:
    - Spring: 31.03.2024 at 02:00 -> 03:00 (skip 02:00-02:59, set load to 0)
    - Autumn: 27.10.2024 at 03:00 -> 02:00 (remove duplicated 02:00-02:59)
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        DataFrame with DST transitions handled
    """
    if 'timestamp' not in df.columns:
        return df
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        return df
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # SPRING 2024: 31.03.2024 - Skip hour 02:00-02:59, set load to 0
    spring_start = pd.Timestamp('2024-03-31 02:00:00')
    spring_end = pd.Timestamp('2024-03-31 03:00:00')
    
    spring_mask = (df_clean['timestamp'] >= spring_start) & (df_clean['timestamp'] < spring_end)
    if spring_mask.any():
        st.info(f"🕐 Sommerzeit-Übergang 2024 (31.03.2024): {spring_mask.sum()} Datenpunkte in der übersprungenen Stunde (02:00-02:59) auf 0 kW gesetzt.")
        # Set load to 0 for the skipped hour
        load_cols = [col for col in df_clean.columns if 'load' in col.lower() or 'kw' in col.lower() or 'value' in col.lower()]
        for col in load_cols:
            if col in df_clean.columns:
                df_clean.loc[spring_mask, col] = 0
    
    # AUTUMN 2024: 27.10.2024 - Remove duplicated hour 02:00-02:59
    autumn_start = pd.Timestamp('2024-10-27 02:00:00')
    autumn_end = pd.Timestamp('2024-10-27 03:00:00')
    
    # Look for duplicated timestamps in the autumn transition period
    autumn_period_mask = (df_clean['timestamp'] >= autumn_start) & (df_clean['timestamp'] < autumn_end)
    
    if autumn_period_mask.any():
        # Find actual duplicates in this period
        autumn_data = df_clean[autumn_period_mask]
        duplicated_mask = autumn_data['timestamp'].duplicated(keep='first')
        
        if duplicated_mask.any():
            # Get indices of duplicated entries to remove
            duplicate_indices = autumn_data[duplicated_mask].index
            st.info(f"🕐 Winterzeit-Übergang 2024 (27.10.2024): {len(duplicate_indices)} doppelte Datenpunkte in der wiederholten Stunde (02:00-02:59) entfernt.")
            df_clean = df_clean.drop(duplicate_indices)
    
    # Remove any remaining duplicates (safety check)
    initial_len = len(df_clean)
    df_clean = df_clean[~df_clean['timestamp'].duplicated(keep='first')]
    removed_duplicates = initial_len - len(df_clean)
    
    if removed_duplicates > 0:
        st.info(f"🔧 {removed_duplicates} zusätzliche doppelte Zeitstempel entfernt.")
    
    # Sort by timestamp to ensure proper order
    df_clean = df_clean.sort_values('timestamp').reset_index(drop=True)
    
    return df_clean