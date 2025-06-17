'''
NAUST.SALTS

Salinity QC work

- Parsing salinometer and cruise log data
- Comparisons between CTD and salinometer
'''

import pandas as pd
import numpy as np
from pathlib import Path

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

def salts_sheet_to_csv(xlsx_file, csv_file):
    '''
    Convert salinometer readings from xlsx sheet to csv.


    '''
    pass


def read_salts_log(xlsx_file):
    '''
    Read a cruise log file (.xlsx) and exctract metadata about the salinity
    samples (sample numbers, Niskin bottle numbers etc). To a Pandas DataFrame.
    '''
    pass


def merge_salts_sheet_and_log(df_salts, sd_log):
    '''
    Combine data from logsheet and salts sheet dataframes into an xarray
    dataset with dimensions (STATION, NISKIN_NUMBER)
    '''

def merge_all_salts_with_btl(ds_salts, ds_btl):
    '''
    Combine data from merged salinometer readings and btl files.

    Also computes salinometer - CTD salinity differences.

    Assumes that STATION and NISKIN_NUMBER dimensions can be combined.
    '''