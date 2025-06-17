'''
NAUST.SALTS

Salinity QC work

- Parsing salinometer and cruise log data
- Comparisons between CTD and salinometer
'''

import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
from kval.data import ctd

def read_salts_sheet(xlsx_file: str | Path) -> pd.DataFrame:
    """
    Parse salinometer readings from Excel sheet to a Pandas DataFrame.

    Parameters:
        xlsx_file (str or Path): Path to the Excel file.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing salinometer readings.
    """
    # Read initial sheet to find "START" marker
    df_preview = pd.read_excel(xlsx_file, header=None)

    # Find the row where 'START' appears in the first column
    try:
        start_row = df_preview[df_preview.iloc[:, 0] == 'START'].index[0]
    except IndexError:
        raise ValueError(f"'START' not found in the first column of {xlsx_file}")

    # Read actual data, starting one row after 'START'
    df = pd.read_excel(xlsx_file, skiprows=start_row + 1)

    # Coerce 'Sample' column to integer, set bad entries to 9999
    if 'Sample' in df.columns:
        df['Sample'] = (
            pd.to_numeric(df['Sample'], errors='coerce', downcast='integer')
            .fillna(9999)
            .astype('int64')
        )
    else:
        raise ValueError("'Sample' column not found in parsed data")

    return df


def salts_sheet_to_csv(xlsx_file: str | Path, csv_file: str | Path):
    """
    Convert salinometer readings from xlsx sheet to csv.

    Parameters:
        xlsx_file (str or Path): Input Excel file
        csv_file (str or Path): Output CSV file path
    """
    df = read_salts_sheet(xlsx_file)
    df.to_csv(csv_file, index=False)



def read_salts_log(xlsx_file: str | Path):
    '''
    Read a cruise log file (.xlsx) and extract metadata about the salinity
    samples (sample numbers, Niskin bottle numbers etc) to a Pandas DataFrame.
    '''

    df_log_all = pd.read_excel(xlsx_file)[
        ['Station', 'CTD LS number', 'Sample name', 'bottle #',
         'Sampling depth (m) from',
         'Sample type', 'Sampling date (UTC)']]

    # Fill NaNs in 'bottle #' before filtering
    df_log_all['NISKIN_NUMBER'] = df_log_all['bottle #'].fillna(9999)

    # Filter only 'Salinity' sample type rows using boolean indexing
    df_log_salt = df_log_all[df_log_all['Sample type'] == 'Salinity'].copy()

    # Convert intended sampling depth to numeric, coercing errors
    df_log_salt['intended_sampling_depth'] = pd.to_numeric(df_log_salt['Sampling depth (m) from'], errors='coerce')

    # Convert NISKIN_NUMBER to int64
    df_log_salt['NISKIN_NUMBER'] = df_log_salt['NISKIN_NUMBER'].astype('int64')

    # Extract digits from 'Sample name' as sample_number, convert safely to int
    df_log_salt['sample_number'] = (
        df_log_salt['Sample name'].str.extract(r'(\d+)', expand=False)
        .astype('int64')
    )

    # Extract digits from Station, zero-pad to 3 chars
    df_log_salt['STATION'] = (
        df_log_salt['Station'].astype(str).str.extract(r'(\d+)', expand=False)
        .str.zfill(3)
    )

    return df_log_salt


def merge_salts_sheet_and_log(
    df_salts: pd.DataFrame,
    df_log: pd.DataFrame
) -> xr.Dataset:
    '''
    Combine salinometer sheet and cruise log DataFrames into an xarray Dataset.

    The resulting Dataset will be indexed by ('STATION', 'NISKIN_NUMBER'),
    containing salinity final values and related metadata.

    Parameters:
        df_salts (pd.DataFrame): Salinometer readings DataFrame (with 'Sample', 'S_final', etc).
        df_log (pd.DataFrame): Cruise log DataFrame (with 'sample_number', 'STATION', 'NISKIN_NUMBER', etc).

    Returns:
        xr.Dataset: Dataset with dimensions STATION and NISKIN_NUMBER.

    Raises:
        ValueError: If required columns are missing or merge results empty DataFrame.
    '''
    required_salts_cols = {'Sample', 'S_final', 'Note'}
    required_log_cols = {'sample_number', 'STATION', 'NISKIN_NUMBER', 'intended_sampling_depth', 'Sampling date (UTC)'}

    missing_salts = required_salts_cols - set(df_salts.columns)
    missing_log = required_log_cols - set(df_log.columns)

    if missing_salts:
        raise ValueError(f"df_salts missing required columns: {missing_salts}")
    if missing_log:
        raise ValueError(f"df_log missing required columns: {missing_log}")

    try:
        df_merged = pd.merge(
            df_log, df_salts,
            left_on='sample_number', right_on='Sample', how='left'
        )
    except Exception as e:
        raise RuntimeError(f"Error during merge: {e}")

    if df_merged.empty:
        raise ValueError("Merge resulted in an empty DataFrame - check input data.")

    cols_to_keep = [
        'intended_sampling_depth', 'Sampling date (UTC)', 'NISKIN_NUMBER',
        'Sample', 'STATION', 'S_final', 'Note'
    ]

    # Check again if all cols to keep are present after merge
    missing_after_merge = set(cols_to_keep) - set(df_merged.columns)
    if missing_after_merge:
        raise ValueError(f"Columns missing after merge: {missing_after_merge}")

    df_merged = df_merged[cols_to_keep]

    df_merged = df_merged.set_index(['STATION', 'NISKIN_NUMBER'])

    ds_merged = xr.Dataset.from_dataframe(df_merged)

    return ds_merged


def merge_all_salts_with_btl(
    ds_salts: xr.Dataset,
    ds_btl: xr.Dataset
) -> xr.Dataset:
    '''
    Combine data from merged salinometer readings and btl files.

    Also computes salinometer - CTD salinity differences.

    Assumes that STATION and NISKIN_NUMBER dimensions can be combined.

    Parameters:
        ds_salts (xr.Dataset): Dataset with salinometer data, dims STATION, NISKIN_NUMBER.
        ds_btl (xr.Dataset): Dataset with bottle CTD data, dims STATION, NISKIN_NUMBER.

    Returns:
        xr.Dataset: Combined Dataset with computed differences and stacked dimension.
    '''

    # Swap TIME to STATION
    if 'TIME' in ds_btl.dims:
        ds_btl = ds_btl.swap_dims({'TIME': 'STATION'})

    # Make a copy of ds_btl to avoid modifying in place
    ds_combined_full = ds_btl.copy()

    # Update with variables from ds_salts (overwrite or add)
    ds_combined_full.update(ds_salts)

    # Ensure all variables have dims in order (STATION, NISKIN_NUMBER)
    for var in ds_combined_full.data_vars:
        if ds_combined_full[var].dims == ('NISKIN_NUMBER', 'STATION'):
            ds_combined_full[var] = ds_combined_full[var].transpose('STATION', 'NISKIN_NUMBER')

    # Check necessary variables exist before computing differences
    if all(x in ds_combined_full.data_vars for x in ['PSAL1', 'S_final']):
        ds_combined_full['Sdiff1'] = ds_combined_full['PSAL1'] - ds_combined_full['S_final']
    else:
        # Could also raise an error or warning here if you want strict checking
        ds_combined_full['Sdiff1'] = xr.full_like(ds_combined_full['S_final'], fill_value=float('nan'))

    if all(x in ds_combined_full.data_vars for x in ['PSAL2', 'S_final']):
        ds_combined_full['Sdiff2'] = ds_combined_full['PSAL2'] - ds_combined_full['S_final']
    else:
        ds_combined_full['Sdiff2'] = xr.full_like(ds_combined_full['S_final'], fill_value=float('nan'))

    # Stack the two dims into one for easier indexing/analysis
    ds_combined = (
        ds_combined_full
        .stack(NISKIN_NUMBER_STATION=['NISKIN_NUMBER', 'STATION'])
        .reset_index('NISKIN_NUMBER_STATION')
    )

    # Drop entries where S_final is NaN
    ds_combined = ds_combined.where(~ds_combined['S_final'].isnull(), drop=True)

    return ds_combined


def build_salts_qc_dataset(
    log_xlsx: Path,
    salts_xlsx: Path,
    btl_dir: Path
) -> xr.Dataset:
    """
    Load and combine salinometer, sample log, and CTD bottle data into a
    single xarray Dataset for quality control and analysis.

    Parameters
    ----------
    log_xlsx : Path
        Path to the sample log sheet (Excel).
    salts_xlsx : Path
        Path to the salinometer lab readings (Excel).
    btl_dir : Path
        Directory containing CTD .btl files.

    Returns
    -------
    xr.Dataset
        Combined dataset with salinometer data, CTD bottle data, and
        computed salinity differences.

    Raises
    ------
    FileNotFoundError
        If any of the input files or directory is missing.
    ValueError
        If required columns or data are missing from the input sheets.
    """
    try:
        df_salts: pd.DataFrame = read_salts_sheet(salts_xlsx)
    except Exception as e:
        raise ValueError(f"Failed to read salinometer sheet '{salts_xlsx}': {e}")

    try:
        df_log: pd.DataFrame = read_salts_log(log_xlsx)
    except Exception as e:
        raise ValueError(f"Failed to read sample log sheet '{log_xlsx}': {e}")

    try:
        ds_salts: xr.Dataset = merge_salts_sheet_and_log(df_salts, df_log)
    except Exception as e:
        raise ValueError(f"Failed to merge salinometer and log data: {e}")

    try:
        ds_btl: xr.Dataset = ctd.dataset_from_btl_dir(btl_dir)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load CTD bottle data from '{btl_dir}': {e}")

    try:
        ds_combined: xr.Dataset = merge_all_salts_with_btl(ds_salts, ds_btl)
    except Exception as e:
        raise ValueError(f"Failed to merge salinometer and CTD datasets: {e}")

    return ds_combined