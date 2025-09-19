"""
NAUST.SALTS

Salinity QC work

This module provides functions to:
- Parse and combine cruise salinometer sheets and sample log sheets into
  xarray Datasets or tidy DataFrames.
- Validate sample numbers and detect duplicates.
- Merge salinity log data with bottle (.btl) files.
- Perform some standardized analysis for salinity QC
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xarray as xr
from kval.data import ctd
from kval.util import xr_funcs
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import mplcursors
from scipy.stats import pearsonr
import warnings

def read_salts_sheet(xlsx_file: str | Path) -> pd.DataFrame:
    """Parse salinometer readings from an Excel sheet into a Pandas DataFrame.

    The function searches for the "START" marker in the first column of the sheet,
    reads the data below it, converts the 'Sample' column to integers (setting
    invalid entries to 9999), renames it to 'SAL_SAMPLE_NUMBER', and checks
    for duplicate sample numbers.

    Args:
        xlsx_file (str | Path): Path to the Excel file containing salinometer data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with salinometer readings and 'SAL_SAMPLE_NUMBER'.

    Raises:
        ValueError: If 'START' is not found in the first column or if the 'Sample'
                    column is missing.
        Exception: If duplicate sample numbers are detected (excluding 9999).
    """
    # Read initial sheet to find "START" marker (ignore warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df_preview = pd.read_excel(xlsx_file, header=None)

    # Find row where 'START' appears
    try:
        start_row = df_preview[df_preview.iloc[:, 0] == 'START'].index[0]
    except IndexError:
        raise ValueError(f"'START' not found in the first column of {xlsx_file}")

    # Read actual data starting one row after 'START' (ignore warnings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        df = pd.read_excel(xlsx_file, skiprows=start_row + 1)

    # Convert 'Sample' column to integer, set bad entries to 9999
    if 'Sample' in df.columns:
        df['Sample'] = (
            pd.to_numeric(df['Sample'], errors='coerce', downcast='integer')
            .fillna(9999)
            .astype('int64')
        )
    else:
        raise ValueError("'Sample' column not found in parsed data")

    # Rename 'Sample' to 'SAL_SAMPLE_NUMBER'
    df_salts = df.rename(columns={"Sample": "SAL_SAMPLE_NUMBER"})

    # Check uniqueness of sample numbers (excluding 9999)
    sample_numbers_are_unique = (
        df_salts.loc[df_salts["SAL_SAMPLE_NUMBER"] != 9999, "SAL_SAMPLE_NUMBER"]
        .is_unique
    )

    if not sample_numbers_are_unique:
        # Find all duplicated sample numbers (excluding 9999)
        duplicates_mask = df_salts["SAL_SAMPLE_NUMBER"].duplicated(keep=False)
        df_duplicates = df_salts.loc[duplicates_mask, "SAL_SAMPLE_NUMBER"]
        df_duplicates = df_duplicates[df_duplicates != 9999]
        duplicates = np.unique(list(df_duplicates))
        raise Exception(
            f'It looks like "{Path(xlsx_file).name}" contains duplicate sample numbers! '
            f'Sample numbers must be unique - correct the sheet?\n'
            f'Duplicate sample numbers: {duplicates}'
        )

    return df_salts




def salts_sheet_to_csv(xlsx_file: str | Path, csv_file: str | Path) -> None:
    """Convert salinometer readings from an Excel sheet to a CSV file.

    Reads the Excel sheet using `read_salts_sheet` and writes the resulting
    DataFrame to a CSV file.

    Args:
        xlsx_file (str | Path): Path to the input Excel file containing salinometer data.
        csv_file (str | Path): Path to the output CSV file.
    """
    df = read_salts_sheet(xlsx_file)
    df.to_csv(csv_file, index=False)


def read_salts_log_tt_style(xlsx_file: str | Path) -> pd.DataFrame:
    """Read a TrollTransect-style cruise log and extract salinity sample metadata.

    This function extracts sample numbers, Niskin bottle numbers, intended sampling
    depth, and station information from a cruise log Excel file, returning a
    cleaned Pandas DataFrame with only salinity samples.

    Args:
        xlsx_file (str | Path): Path to the cruise log Excel file.

    Returns:
        pd.DataFrame: DataFrame containing salinity samples with columns:
            - STATION (str): Zero-padded station number.
            - SAL_SAMPLE_NUMBER (int): Extracted sample number from 'Sample name'.
            - NISKIN_NUMBER (int): Bottle number (9999 if missing).
            - intended_sampling_depth (float): Sampling depth in meters.
            - Other relevant columns from the original log.
    """
    df_log_all = pd.read_excel(xlsx_file)[
        ['Station', 'CTD LS number', 'Sample name', 'bottle #',
         'Sampling depth (m) from',
         'Sample type', 'Sampling date (UTC)']
    ]

    # Fill NaNs in 'bottle #' before filtering
    df_log_all['NISKIN_NUMBER'] = df_log_all['bottle #'].fillna(9999)

    # Filter only 'Salinity' sample type rows
    df_log_salt = df_log_all[df_log_all['Sample type'] == 'Salinity'].copy()

    # Convert intended sampling depth to numeric
    df_log_salt['intended_sampling_depth'] = pd.to_numeric(
        df_log_salt['Sampling depth (m) from'], errors='coerce'
    )

    # Convert NISKIN_NUMBER to int64
    df_log_salt['NISKIN_NUMBER'] = df_log_salt['NISKIN_NUMBER'].astype('int64')

    # Extract digits from 'Sample name' as SAL_SAMPLE_NUMBER
    df_log_salt['SAL_SAMPLE_NUMBER'] = (
        df_log_salt['Sample name'].str.extract(r'(\d+)', expand=False)
        .astype('int64')
    )

    # Extract digits from Station, zero-pad to 3 chars
    df_log_salt['STATION'] = (
        df_log_salt['Station'].astype(str).str.extract(r'(\d+)', expand=False)
        .str.zfill(3)
    )

    return df_log_salt



def read_salts_log_fs_style(xlsx_file: str) -> xr.Dataset:
    """Read a Fram Strait-style cruise log and extract salinity sample metadata.

    This function reads a cruise log Excel sheet, extracts station-level metadata
    and salinity sample numbers, and converts the information into an xarray
    Dataset. The resulting dataset has dimensions 'STATION' and 'NISKIN_NUMBER',
    with 'SAL_SAMPLE_NUMBER' as the primary data variable and additional
    metadata stored as station-level coordinates.

    Args:
        xlsx_file (str): Path to the Fram Strait log Excel file.

    Returns:
        pd.DataFrame: Flattened DataFrame with the following columns:
            - STATION (str): Station identifier.
            - NISKIN_NUMBER (int): Bottle number.
            - SAL_SAMPLE_NUMBER (int): Sample number extracted from the sheet.
            - Additional station-level metadata as available.
    """
    # --- Read the log sheet ---
    df_log_all = pd.read_excel(xlsx_file)

    # --- Extract header info ---
    cut_idx = df_log_all.index[df_log_all["Unnamed: 0"] == "Moon Pool? yes/no (0/1)"][0]
    df_cut = df_log_all.iloc[:cut_idx + 1, :]
    df_station_meta = (
        df_cut.set_index("Unnamed: 0")
        .T
        .rename_axis("STN")
        .reset_index()
    )
    df_station_meta = df_station_meta[df_station_meta["STN"] != "STN"].copy()
    df_station_meta["STN"] = pd.to_numeric(df_station_meta["STN"], errors="coerce")
    df_station_meta = df_station_meta.rename(columns={"STN": "STATION"})

    # --- Extract salinity sample information ---
    sal_start_idx = df_log_all.index[df_log_all["Unnamed: 0"] == "Salinity"][0]
    mask = df_log_all.loc[sal_start_idx + 1:, "STN"].isna()
    sal_end_idx = mask.idxmax() if mask.any() else len(df_log_all)
    df_sal = df_log_all.iloc[sal_start_idx:sal_end_idx, :]

    # Clean up salinity DataFrame
    df_sal = df_sal.iloc[:, 1:].copy()  # drop leftmost column
    df_sal = df_sal.rename(columns={"STN": "NISKIN_NUMBER"})
    df_sal["NISKIN_NUMBER"] = df_sal["NISKIN_NUMBER"].astype(int)

    # Read stations
    station_list = list(df_sal.columns.drop("NISKIN_NUMBER"))

    # Convert to xarray Dataset
    ds_sal = xr.Dataset(
        data_vars={
            "SAL_SAMPLE_NUMBER": (
                ("NISKIN_NUMBER", "STATION"),
                df_sal.to_numpy()[:, 1:]
            )
        },
        coords={
            "STATION": station_list,
            "NISKIN_NUMBER": df_sal["NISKIN_NUMBER"]
        }
    )

    # Add station-level metadata (pressure, echo depth, etc.)
    for meta_var in df_station_meta.columns:
        if meta_var != "STATION":
            ds_sal[meta_var] = ("STATION", df_station_meta[meta_var])

    # Convert to flattened DataFrame
    df_sal = (
        ds_sal
        .to_dataframe()        # MultiIndex DataFrame (NISKIN_NUMBER, STATION)
        .reset_index()         # Flatten MultiIndex
        .dropna(subset=["SAL_SAMPLE_NUMBER"])  # Remove rows with missing samples
        .astype(int)           # Convert sample numbers to integer
    )

    return df_sal


def read_salts_log(xlsx_file: str) -> xr.Dataset:
    '''
    Read an cruise sample log sheet and convert it to an xarray Dataset.

    Tries to read the TrollTransect format, if that fails tries to read the Fram 
    Strait format.

    '''

    try:
        df_sal = read_salts_log_tt_style(xlsx_file)
    except:
        try:
            df_sal = read_salts_log_fs_style(xlsx_file)
        except:
            raise Exception(f'Unable to parse {xlsx_file}')
    
    try:
        df_sal["STATION"] = df_sal["STATION"].astype(str).str.lstrip("0").astype(int)
    except:
        warnings.warn('Unable to convert all values of "station" in the .btl files -'
                      ' probably non-numerical values. Correct or proceed with caution.',
                      UserWarning)

    return df_sal


def merge_salts_sheet_and_log(
    df_salts: pd.DataFrame,
    df_log: pd.DataFrame
) -> xr.Dataset:
    """Combine salinometer sheet and cruise log DataFrames into an xarray Dataset.

    The resulting Dataset is indexed by ('STATION', 'NISKIN_NUMBER') and contains
    salinity values and associated metadata.

    Args:
        df_salts (pd.DataFrame): Salinometer readings DataFrame with columns such as:
            - SAL_SAMPLE_NUMBER
            - S_final
            - Note
        df_log (pd.DataFrame): Cruise log DataFrame with columns such as:
            - SAL_SAMPLE_NUMBER
            - STATION
            - NISKIN_NUMBER

    Returns:
        xr.Dataset: Dataset with dimensions STATION and NISKIN_NUMBER. The variable
            'S_final' is renamed to 'PSAL_LAB'.

    Raises:
        ValueError: If required columns are missing in either DataFrame or if the merge
            results in an empty DataFrame.
        RuntimeError: If an unexpected error occurs during merging.
    """
    required_salts_cols = {'SAL_SAMPLE_NUMBER', 'S_final', 'Note'}
    required_log_cols = {'SAL_SAMPLE_NUMBER', 'STATION', 'NISKIN_NUMBER'}

    missing_salts = required_salts_cols - set(df_salts.columns)
    missing_log = required_log_cols - set(df_log.columns)

    if missing_salts:
        raise ValueError(f"df_salts missing required columns: {missing_salts}")
    if missing_log:
        raise ValueError(f"df_log missing required columns: {missing_log}")

    try:
        df_merged = pd.merge(
            df_log, df_salts,
            left_on='SAL_SAMPLE_NUMBER', right_on='SAL_SAMPLE_NUMBER', how='left'
        )
    except Exception as e:
        raise RuntimeError(f"Error during merge: {e}")

    if df_merged.empty:
        raise ValueError("Merge resulted in an empty DataFrame - check input data.")

    cols_to_keep = [
        'NISKIN_NUMBER', 'SAL_SAMPLE_NUMBER', 'STATION', 'S_final', 'Note'
    ]
    missing_after_merge = set(cols_to_keep) - set(df_merged.columns)
    if missing_after_merge:
        raise ValueError(f"Columns missing after merge: {missing_after_merge}")

    # Set multi-index and convert to xarray Dataset
    df_merged = df_merged.set_index(['STATION', 'NISKIN_NUMBER'])
    ds_merged = xr.Dataset.from_dataframe(df_merged)

    # Rename salinity variable for clarity
    ds_merged = ds_merged.rename_vars({'S_final': 'PSAL_LAB'})

    return ds_merged



def merge_all_salts_with_btl(
    ds_salts: xr.Dataset,
    ds_btl: xr.Dataset,
    stack_to_1D: bool = True
) -> xr.Dataset:
    """Combine merged salinometer readings with bottle (CTD) data.

    Computes differences between salinometer and CTD salinity and optionally
    stacks the STATION and NISKIN_NUMBER dimensions into a single 1D dimension.

    Args:
        ds_salts (xr.Dataset): Dataset containing salinometer data with dimensions
            ('STATION', 'NISKIN_NUMBER'), including variable 'PSAL_LAB'.
        ds_btl (xr.Dataset): Dataset containing bottle CTD data with dimensions
            ('STATION', 'NISKIN_NUMBER'), including variables like 'PSAL1', 'PSAL2'.
        stack_to_1D (bool): If True, stack ('STATION', 'NISKIN_NUMBER') into a single
            dimension 'NISKIN_NUMBER_STATION'. Default is True.

    Returns:
        xr.Dataset: Combined Dataset including:
            - Original CTD variables
            - Salinometer data ('PSAL_LAB')
            - Differences ('Sdiff1', 'Sdiff2')
            - Optionally stacked dimension for easier indexing

    Notes:
        - If 'TIME' is a dimension in ds_btl, it is swapped to 'STATION'.
        - If a necessary variable is missing, the difference variables are filled
          with NaNs.
    """
    # Swap TIME dimension to STATION if present
    if 'TIME' in ds_btl.dims:
        ds_btl = xr_funcs.swap_var_coord(ds_btl, 'TIME', 'STATION')

    # Convert NISKIN_NUMBER to int
    ds_btl["NISKIN_NUMBER"] = ds_btl["NISKIN_NUMBER"].astype(int)

    # Convert STATION to int (strip leading zeros)
    try:
        ds_btl["STATION"] = ds_btl["STATION"].astype(str).str.lstrip("0").astype(int)
    except Exception:
        warnings.warn(
            'Unable to convert all values of "STATION" in ds_btl; '
            'probably non-numerical values. Proceed with caution.',
            UserWarning
        )

    # Copy to avoid modifying in place
    ds_combined_full = ds_btl.copy()

    # Update/add variables from ds_salts
    ds_combined_full.update(ds_salts)

    # Ensure all variables have dimensions ordered as ('STATION', 'NISKIN_NUMBER')
    for var in ds_combined_full.data_vars:
        if ds_combined_full[var].dims == ('NISKIN_NUMBER', 'STATION'):
            ds_combined_full[var] = ds_combined_full[var].transpose('STATION', 'NISKIN_NUMBER')

    # Compute salinity differences
    if all(var in ds_combined_full.data_vars for var in ['PSAL1', 'PSAL_LAB']):
        ds_combined_full['Sdiff1'] = ds_combined_full['PSAL1'] - ds_combined_full['PSAL_LAB']
    else:
        ds_combined_full['Sdiff1'] = xr.full_like(ds_combined_full['PSAL_LAB'], fill_value=float('nan'))

    if all(var in ds_combined_full.data_vars for var in ['PSAL2', 'PSAL_LAB']):
        ds_combined_full['Sdiff2'] = ds_combined_full['PSAL2'] - ds_combined_full['PSAL_LAB']
    else:
        ds_combined_full['Sdiff2'] = xr.full_like(ds_combined_full['PSAL_LAB'], fill_value=float('nan'))

    # Optionally stack dimensions
    if stack_to_1D:
        ds_combined = (
            ds_combined_full
            .stack(NISKIN_NUMBER_STATION=['NISKIN_NUMBER', 'STATION'])
            .reset_index('NISKIN_NUMBER_STATION')
        )
        # Drop rows where PSAL_LAB is NaN
        ds_combined = ds_combined.where(~ds_combined['PSAL_LAB'].isnull(), drop=True)
    else:
        ds_combined = ds_combined_full

    return ds_combined

def build_salts_qc_dataset(
    log_xlsx: str | Path,
    salts_xlsx: str | Path,
    btl_dir: str | Path
) -> xr.Dataset:
    """
    Load and combine salinometer, sample log, and CTD bottle data into a single xarray Dataset.

    This function reads the salinometer Excel sheet, the cruise sample log
    (TrollTransect or Fram Strait style), and all CTD bottle files in a
    directory. It merges these datasets and computes salinity differences
    for quality control.

    Args:
        log_xlsx (str or Path): Path to the sample log sheet (Excel).
        salts_xlsx (str or Path): Path to the salinometer lab readings (Excel).
        btl_dir (str or Path): Directory containing CTD .btl files.

    Returns:
        xr.Dataset: Combined dataset including:
            - Salinometer measurements
            - CTD bottle data
            - Computed differences (salinometer minus CTD)

    Raises:
        FileNotFoundError: If any input file or directory is missing.
        ValueError: If required columns or data are missing from input sheets
            or if merges fail.

    Example:
        >>> from pathlib import Path
        >>> import xarray as xr
        >>> ds = build_salts_qc_dataset(
        ...     log_xlsx="/data/cruise_log.xlsx",
        ...     salts_xlsx="/data/salinity_lab.xlsx",
        ...     btl_dir="/data/btl_files"
        ... )
        >>> ds
        <xarray.Dataset>
        Dimensions:  (NISKIN_NUMBER_STATION: 120)
        Coordinates:
            STATION   (NISKIN_NUMBER_STATION) int64 ...
            NISKIN_NUMBER  (NISKIN_NUMBER_STATION) int64 ...
        Data variables:
            PSAL_LAB   (NISKIN_NUMBER_STATION) float64 ...
            Sdiff1     (NISKIN_NUMBER_STATION) float64 ...
            Sdiff2     (NISKIN_NUMBER_STATION) float64 ...
    """
    # Convert all inputs to Path objects
    log_xlsx = Path(log_xlsx)
    salts_xlsx = Path(salts_xlsx)
    btl_dir = Path(btl_dir)

    # Read salinometer sheet
    try:
        df_salts: pd.DataFrame = read_salts_sheet(salts_xlsx)
    except Exception as e:
        raise ValueError(f"Failed to read salinometer sheet '{salts_xlsx}': {e}")

    # Read cruise sample log (TrollTransect or Fram Strait)
    try:
        df_log: pd.DataFrame | xr.Dataset = read_salts_log(log_xlsx)
    except Exception as e:
        raise ValueError(f"Failed to read sample log sheet '{log_xlsx}': {e}")

    # Merge salinometer sheet with log
    try:
        ds_salts: xr.Dataset = merge_salts_sheet_and_log(df_salts, df_log)
    except Exception as e:
        raise ValueError(f"Failed to merge salinometer and log data: {e}")

    # Load CTD bottle datasets (keep as Path)
    try:
        ds_btl: xr.Dataset = ctd.dataset_from_btl_dir(btl_dir)
    except Exception as e:
        raise FileNotFoundError(f"Failed to load CTD bottle data from '{btl_dir}': {e}")

    # Merge salinometer data with CTD bottles
    try:
        ds_combined: xr.Dataset = merge_all_salts_with_btl(ds_salts, ds_btl)
    except Exception as e:
        raise ValueError(f"Failed to merge salinometer and CTD datasets: {e}")

    return ds_combined



def plot_salinity_diff_histogram(
    ds: xr.Dataset,
    psal_var: str | None = None,
    salinometer_var: str = "PSAL_LAB",
    min_pres: float = 500,
    x_range: bool | tuple = (-0.012, 0.012),
    N: int = 24,
    figsize: tuple = (4, 3.5),
):
    """
    Plot a histogram of salinity differences between a CTD variable and salinometer readings.

    The function calculates the difference between a CTD salinity variable and the
    salinometer measurements, filters by a minimum pressure, and displays a histogram
    with mean, median, and standard deviation information.

    Args:
        ds (xr.Dataset): Dataset containing CTD and salinometer data.
        psal_var (str, optional): Name of the CTD salinity variable (e.g., 'PSAL1', 'PSAL2').
            If None, the function will auto-detect a suitable variable.
        salinometer_var (str, default 'PSAL_LAB'): Name of the salinometer salinity variable.
        min_pres (float, default 500): Minimum pressure to include in the comparison.
        x_range (tuple or None, default (-0.012, 0.012)): The lower and upper range of the bins.
        N (int, default 24): Number of bins in the histogram.
        figsize (tuple, default (10, 3.5)): Size of the figure in inches.

    Raises:
        ValueError: If required variables are missing from the dataset or if no valid
            data points remain after filtering by pressure.

    Returns:
        None: Displays a histogram plot in Jupyter.
    """
    # Auto-detect PSAL variable if not provided
    if psal_var is None:
        for default_var in ["PSAL1", "PSAL", "PSAL2"]:
            if default_var in ds:
                psal_var = default_var
                break
        else:
            raise ValueError("No PSAL variable found. Specify `psal_var` explicitly.")

    # Validate presence of variables
    for var in [salinometer_var, "PRES"]:
        if var not in ds:
            raise ValueError(f"Required variable '{var}' not found in dataset.")

    # Compute difference and filter by pressure
    SAL_diff = ds[psal_var] - ds[salinometer_var]
    deep = SAL_diff.where(ds.PRES > min_pres).astype(float)
    valid = deep.where(~deep.isnull(), drop=True)

    if valid.size == 0:
        raise ValueError("No valid data points after pressure filtering.")

    # Compute statistics
    diff_mean = valid.mean().item()
    diff_median = valid.median().item()
    diff_std = valid.std().item()
    count = valid.size
    sem = diff_std / np.sqrt(count)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(valid.values.flatten(), range = x_range, bins=N, 
            color="steelblue", alpha=0.7, zorder = 2,)
    # Plot outline as line for emphasis
    ax.hist(valid.values.flatten(), range = x_range, bins=N, 
        color="k", alpha=0.7, zorder = 2,
       histtype = 'step', lw = 0.6)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Lines for mean, median, and zero
    ax.axvline(0, color="k", ls=":", lw=0.7)
    mean_line = ax.axvline(diff_mean, color="tab:orange", dashes=(5, 3), lw=1.5,
                            label=f"Mean:\n{diff_mean:.4f}", zorder = 2)
    median_line = ax.axvline(diff_median, color="tab:orange", ls=":", lw=2,
                              label=f"Median:\n{diff_median:.4f}", zorder = 2)

    # Labels and title
    ax.set_xlabel(f"{psal_var} - {salinometer_var}")
    ax.set_ylabel("# SAMPLES")
    ax.set_title(f"{psal_var}: Salinity difference ",
                fontsize = 9.5)
    

    ax.text(1.001, 0.96, f'n={count}',
            rotation = -90, color = 'steelblue', transform=ax.transAxes, 
            fontweight = 'bold', ha="left", va="top")
    if min_pres > 0:
        pres_text = f'PRES > {min_pres} dbar'
    else:
        pres_text = 'All samples'
        
    ax.text(1.001, 0, pres_text,
        rotation = -90, color = 'gray', transform=ax.transAxes, 
        fontweight = 'normal', ha="left", va="bottom")
   
    # Dummy handle for standard deviation in legend
    std_handle = mlines.Line2D([], [], color="none", label=f"Std=\n{diff_std:.4f}")

    # Shade plus/minus 0.003
    shade = ax.axvspan(-0.003, 0.003, color='tab:green', alpha=0.15, zorder = 0.5, label = '±0.003')

    # Legend   
    leg = ax.legend(handles=[mean_line, median_line, std_handle, shade], fontsize = 8.5,
                   handlelength = 0.9)
    
    plt.tight_layout()
    ax.grid(True, zorder = 0)

    # Jupyter interactive close button
    try:
        button = widgets.Button(description="Close", layout=widgets.Layout(width="150px"))
        display(button)

        def close_fig(_):
            plt.close(fig)
            button.close()

        button.on_click(close_fig)
    except Exception:
        pass
        

def plot_by_sample(
    ds: xr.Dataset,
    psal_var: str = "PSAL1",
    salinometer_var: str = "PSAL_LAB",
    SAL_SAMPLE_NUMBER_var: str = "SAL_SAMPLE_NUMBER",
    min_pres: float = 0,
):
    """
    Plot salinity comparison by sample number for CTD vs salinometer measurements.

    The function plots three panels:
        1. Salinity values for CTD and salinometer per sample.
        2. Salinity difference (CTD - salinometer) per sample.
        3. Histogram of salinity differences.

    Interactive hover annotations and a close button are included (Jupyter only).

    Args:
        ds (xr.Dataset): Dataset containing CTD and salinometer data.
        psal_var (str, default 'PSAL1'): Name of the CTD salinity variable.
        salinometer_var (str, default 'PSAL_LAB'): Name of the salinometer variable.
        SAL_SAMPLE_NUMBER_var (str, default 'SAL_SAMPLE_NUMBER'): Name of the sample number variable.
        min_pres (float, default 0): Minimum pressure to include in the analysis.

    Raises:
        ValueError: If required variables are missing from the dataset.

    Returns:
        None: Displays a multi-panel plot in Jupyter.
    """
    # Validate required variables
    for var in [psal_var, salinometer_var, SAL_SAMPLE_NUMBER_var, "PRES"]:
        if var not in ds:
            raise ValueError(f"Dataset missing required variable: {var}")

    # Filter by minimum pressure
    mask = ds.PRES > min_pres
    b = xr.Dataset(coords={"NISKIN_NUMBER": ds.get("NISKIN_NUMBER"),
                           "STATION": ds.get("STATION")})
    b[psal_var] = ds[psal_var].where(mask)
    b[salinometer_var] = ds[salinometer_var].where(mask)
    b[SAL_SAMPLE_NUMBER_var] = ds[SAL_SAMPLE_NUMBER_var].where(mask)
    b["PRES"] = ds.PRES.where(mask)

    # Flatten arrays
    psal_vals = b[psal_var].values.flatten()
    salinometer_vals = b[salinometer_var].values.flatten()
    sample_nums = b[SAL_SAMPLE_NUMBER_var].values.flatten()
    pres_vals = b["PRES"].values.flatten()

    # Compute differences and statistics
    sal_diff = (psal_vals - salinometer_vals).astype(float)
    N_count = np.count_nonzero(~np.isnan(sal_diff))
    Sdiff_mean = np.nanmean(sal_diff)

    # Sort by sample number (ignoring NaNs)
    valid_mask = ~np.isnan(sample_nums) & ~np.isnan(sal_diff)
    sorted_indices = np.argsort(sample_nums[valid_mask])
    sample_num_sorted = sample_nums[valid_mask][sorted_indices]
    sal_diff_sorted = sal_diff[valid_mask][sorted_indices]
    pres_sorted = pres_vals[valid_mask][sorted_indices]

    # Labels for interactive annotation
    point_labels = [f"Sample #{sn:.0f} ({p:.0f} dbar)" 
                    for sn, p in zip(sample_num_sorted, pres_sorted)]

    # Create figure and axes
    fig = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

    # Panel 1: salinity values
    ax0.plot(sample_nums, psal_vals, '.', color="tab:blue", lw=0.2, alpha=0.6,
             label=f"Bottle file {psal_var}", zorder=2)
    ax0.plot(sample_nums, salinometer_vals, '.', color="tab:orange", lw=0.2, alpha=0.6,
             label="Salinometer", zorder=2)
    ax0.set_xlabel("SAMPLE NUMBER")
    ax0.set_ylabel("Practical salinity")
    ax0.grid(True)

    # Correlation coefficient
    valid_corr_mask = ~np.isnan(psal_vals) & ~np.isnan(salinometer_vals)
    if np.count_nonzero(valid_corr_mask) > 1:
        r, pval = pearsonr(psal_vals[valid_corr_mask], salinometer_vals[valid_corr_mask])
        corr_text = f"r = {r:.3f}, p = {pval:.1e}"
    else:
        corr_text = "r = NaN, p = NaN"
    ax0.text(0.02, 0.95, corr_text, transform=ax0.transAxes, fontsize=10,
             verticalalignment="top",
             bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    ax0.legend()

    # Interactive hover annotations
    cursor = mplcursors.cursor(ax0.collections, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        ind = sel.index
        if ind < len(point_labels):
            sel.annotation.set_text(point_labels[ind])

    # Panel 2: salinity difference
    ax1.fill_between(sample_num_sorted, sal_diff_sorted, color="k", alpha=0.3,
                     label="Bottle file", lw=0.2, zorder=2)
    ax1.plot(sample_num_sorted, sal_diff_sorted, '.', color="tab:red", alpha=0.8,
             label="Salinometer", lw=0.2, zorder=2)
    ax1.axhline(Sdiff_mean, color="tab:blue", lw=1.6, alpha=0.75, ls=":",
                label=f"Mean = {Sdiff_mean:.2e}")
    ax1.set_xlabel("SAMPLE NUMBER")
    ax1.set_ylabel(f"{psal_var} - salinometer S")
    ax1.grid(True)
    ax1.legend()

    # Panel 3: histogram
    ax2.hist(sal_diff, bins=20, orientation="horizontal", color="tab:red", alpha=0.7)
    ax2.set_ylim(ax1.get_ylim())
    ax2.axhline(0, color="k", ls="--")
    ax2.axhline(Sdiff_mean, color="tab:blue", lw=1.6, alpha=0.75, ls=":",
                label=f"Mean = {Sdiff_mean:.2e}")
    ax2.set_xlabel("# SAMPLES")
    ax2.set_ylabel(f"{psal_var} - salinometer S")
    ax2.grid(True)
    ax2.legend(loc=0, bbox_to_anchor=(1, 1.2), fontsize=9)

    fig.suptitle(f"Salinity comparison for samples taken at >{min_pres} dbar (n = {N_count})")
    plt.tight_layout()

    # Interactive close button (Jupyter)
    def close_everything(_):
        plt.close(fig)
        button_exit.close()

    button_exit = widgets.Button(description="Close", layout=widgets.Layout(width="200px"))
    button_exit.on_click(close_everything)
    display(button_exit)


def plot_scatter(
    ds: xr.Dataset,
    psal_var: str = "PSAL1",
    salinometer_var: str = "PSAL_LAB",
    SAL_SAMPLE_NUMBER_var: str = "SAL_SAMPLE_NUMBER",
    min_pres: float = 0,
    show_corr: bool = True
):
    """
    Plot a scatter comparison of CTD salinity vs salinometer salinity.

    Filters by minimum pressure and displays a 1:1 reference line. Optionally
    computes and annotates the correlation coefficient.

    Args:
        ds (xr.Dataset): Dataset containing CTD and salinometer data.
        psal_var (str, default 'PSAL1'): Name of the CTD salinity variable.
        salinometer_var (str, default 'PSAL_LAB'): Name of the salinometer variable.
        SAL_SAMPLE_NUMBER_var (str, default 'SAL_SAMPLE_NUMBER'): Name of the sample number variable.
        min_pres (float, default 0): Minimum pressure to include in the plot.
        show_corr (bool, default True): Whether to display correlation coefficient on plot.

    Raises:
        ValueError: If required variables are missing or no PSAL variable is found.

    Returns:
        None: Displays a scatter plot with equal axes.
    """
    # Auto-detect psal_var if not provided
    if psal_var is None:
        for default_var in ['PSAL1', 'PSAL', 'PSAL2']:
            if default_var in ds:
                psal_var = default_var
                break
        else:
            raise ValueError("No PSAL variable found. Specify `psal_var` explicitly.")

    # Validate variables
    for var in [salinometer_var, 'PRES']:
        if var not in ds:
            raise ValueError(f"Dataset missing required variable: {var}")

    # Filter by minimum pressure
    ds_deep = ds.where(ds.PRES > min_pres, drop = True)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_aspect('equal')
    ax.plot(ds_deep[psal_var], ds_deep[salinometer_var], '.', zorder=2)
    
    # Determine axis limits and 1:1 line
    xl, yl = ax.get_xlim(), ax.get_ylim()
    min_range = min(xl[0], yl[0])
    max_range = max(xl[1], yl[1])
    ax.plot([min_range, max_range], [min_range, max_range], 
            'k', dashes = (5, 5), lw=0.5, zorder=0, label = '1:1')
    ax.set_xlim(min_range, max_range)
    ax.set_ylim(min_range, max_range)
    ax.set_xticks(ax.get_yticks())
    plt.xticks(rotation=90)

    # Axis labels
    ax.set_xlabel(psal_var)
    ax.set_ylabel(salinometer_var)

    # Optional correlation coefficient
    if show_corr:
        corr = xr.corr(ds_deep[psal_var], ds_deep[salinometer_var]).data
        ax.text(0.02, 0.96, f'Correlation: {corr:.3f}',
                transform=ax.transAxes, ha="left", va="top")

    # Optional title
    if min_pres > 0:
        ax.set_title(f'Samples from >{min_pres} dbar')

    ax.text(1.001, 0.96, f'n={len(ds_deep[psal_var])}',
            rotation = -90, color = 'steelblue', transform=ax.transAxes, 
            fontweight = 'bold', ha="left", va="top")
    
    ax.legend(fontsize = 10)
    ax.grid()
    plt.tight_layout()
