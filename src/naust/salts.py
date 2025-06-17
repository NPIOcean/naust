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
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import mplcursors
from scipy.stats import pearsonr

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
        df_salts (pd.DataFrame): Salinometer readings DataFrame (with 'Sample',
        'S_final', etc). df_log (pd.DataFrame): Cruise log DataFrame (with
        'sample_number', 'STATION', 'NISKIN_NUMBER', etc).

    Returns:
        xr.Dataset: Dataset with dimensions STATION and NISKIN_NUMBER.

    Raises:
        ValueError: If required columns are missing or merge results empty
        DataFrame.
    '''
    required_salts_cols = {'Sample', 'S_final', 'Note'}
    required_log_cols = {'sample_number', 'STATION', 'NISKIN_NUMBER',
                         'intended_sampling_depth', 'Sampling date (UTC)'}

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


    # Rename S_final to PSAL_LAB
    ds_merged = ds_merged.rename_vars({'S_final': 'PSAL_LAB'})
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
    if all(x in ds_combined_full.data_vars for x in ['PSAL1', 'PSAL_LAB']):
        ds_combined_full['Sdiff1'] = ds_combined_full['PSAL1'] - ds_combined_full['PSAL_LAB']
    else:
        # Could also raise an error or warning here if you want strict checking
        ds_combined_full['Sdiff1'] = xr.full_like(ds_combined_full['PSAL_LAB'], fill_value=float('nan'))

    if all(x in ds_combined_full.data_vars for x in ['PSAL2', 'PSAL_LAB']):
        ds_combined_full['Sdiff2'] = ds_combined_full['PSAL2'] - ds_combined_full['PSAL_LAB']
    else:
        ds_combined_full['Sdiff2'] = xr.full_like(ds_combined_full['PSAL_LAB'], fill_value=float('nan'))


    # Stack the two dims into one for easier indexing/analysis
    ds_combined = (
        ds_combined_full
        .stack(NISKIN_NUMBER_STATION=['NISKIN_NUMBER', 'STATION'])
        .reset_index('NISKIN_NUMBER_STATION')
    )

    # Drop entries where PSAL_LAB is NaN
    ds_combined = ds_combined.where(~ds_combined['PSAL_LAB'].isnull(), drop=True)

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



def plot_salinity_diff_histogram(
    ds: xr.Dataset,
    psal_var: str = None,
    salinometer_var: str = "PSAL_LAB",
    min_pres: float = 500,
    N: int = 20,
    figsize=(10, 3.5)
):
    """
    Plot a histogram of the salinity difference between a CTD variable and the salinometer.

    Parameters:
    ----------
    ds : xr.Dataset
        Dataset returned by `build_salts_qc_dataset`, containing CTD and salinometer data.
    psal_var : str, optional
        Name of CTD salinity variable (e.g., 'PSAL1', 'PSAL2').
        If None, will auto-detect.
    salinometer_var : str, default 'PSAL_LAB'
        Name of the salinometer salinity variable.
    min_pres : float, default 500
        Minimum pressure to include in the comparison.
    N : int, default 20
        Number of histogram bins.
    figsize : tuple, default (7, 3.5)
        Size of figure in inches.

    Returns:
    -------
    None. Displays the plot in Jupyter.
    """

    if psal_var is None:
        for default_var in ['PSAL1', 'PSAL', 'PSAL2']:
            if default_var in ds:
                psal_var = default_var
                break
        else:
            raise ValueError("No PSAL variable found. Specify `psal_var` explicitly.")

    if salinometer_var not in ds:
        raise ValueError(f"Salinometer variable '{salinometer_var}' not found in dataset.")

    if 'PRES' not in ds:
        raise ValueError("Pressure variable 'PRES' not found in dataset.")

    # Compute the difference
    SAL_diff = ds[psal_var] - ds[salinometer_var]

    # Filter to deep samples
    deep = SAL_diff.where(ds.PRES > min_pres).astype(float)

    # Statistics
    valid = deep.where(~deep.isnull(), drop=True)
    if valid.size == 0:
        raise ValueError("No valid data points after pressure filtering.")

    diff_mean = valid.mean().item()
    diff_median = valid.median().item()
    count = valid.size

    diff_std = valid.std().values
    sem = diff_std / np.sqrt(count)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.hist(valid.values.flatten(), bins=N, color='steelblue', alpha=0.7)

    ax.axvline(0, color='k', ls='--', lw=1)
    mean_line = ax.axvline(diff_mean, color='tab:red', dashes=(5, 3), lw=1,
               label=f'Mean = {diff_mean:.4f}')
    median_line = ax.axvline(diff_median, color='tab:red', ls=':', lw=1.5,
               label=f'Median = {diff_median:.4f}')

    ax.set_xlabel(f"{psal_var} - {salinometer_var}")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{psal_var}: Salinity difference at pressure > {min_pres} dbar (n={count})")
    ax.grid(True)
    #ax.legend()



    # Create dummy (invisible) handle for std
    std_handle = mlines.Line2D([], [], color='none', label=f'Std = {diff_std:.4f}')

    # Now create legend with both mean/median lines and dummy std handle
    ax.legend(handles=[mean_line, median_line, std_handle], loc = 1,
              bbox_to_anchor = (1.3, 0.6))

    plt.tight_layout()


    # Close button (Jupyter only)
    try:
        button = widgets.Button(description="Close", layout=widgets.Layout(width='150px'))
        display(button)

        def close_fig(_):
            plt.close(fig)
            button.close()

        button.on_click(close_fig)
    except Exception:
        pass



def plot_by_sample(ds, psal_var='PSAL1', salinometer_var='PSAL_LAB',
                   sample_number_var='Sample', min_pres=0):
    """
    Plot salinity comparison for samples taken at depths greater than a specified minimum pressure,
    organized by sample number.

    Parameters:
    - ds (xr.Dataset): Input dataset containing salinity variables.
    - psal_var (str): Name of the salinity variable to compare with PSAL_LAB.
                      Defaults to 'PSAL1'.
    - salinometer_var (str): Name of the salinometer salinity variable.
    - sample_number_var (str): Name of the sample number variable.
    - min_pres (float): Minimum pressure threshold for samples. Defaults to 500.

    Returns:
    None

    Displays a multi-panel plot with interactive annotations and a close button.
    """

    # Check required variables in ds
    for var in [psal_var, salinometer_var, sample_number_var, 'PRES']:
        if var not in ds:
            raise ValueError(f"Dataset missing required variable: {var}")

    # Mask data where PRES > min_pres
    mask = ds.PRES > min_pres

    # Build filtered dataset b (only where mask is True)
    b = xr.Dataset(coords={'NISKIN_NUMBER': ds.get('NISKIN_NUMBER'),
                          'STATION': ds.get('STATION')})
    b[psal_var] = ds[psal_var].where(mask)
    b[salinometer_var] = ds[salinometer_var].where(mask)
    b[sample_number_var] = ds[sample_number_var].where(mask)
    b['PRES'] = ds.PRES.where(mask)

    # Flatten arrays for convenience
    psal_vals = b[psal_var].values.flatten()
    salinometer_vals = b[salinometer_var].values.flatten()
    sample_nums = b[sample_number_var].values.flatten()
    pres_vals = b['PRES'].values.flatten()

    # Calculate salinity difference and stats
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
    point_labels = [
        f"Sample #{sn:.0f} ({p:.0f} dbar)"
        for sn, p in zip(sample_num_sorted, pres_sorted)
    ]

    # Create figure and axes
    fig = plt.figure(figsize=(10, 6))
    ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
    ax1 = plt.subplot2grid((2, 4), (1, 0), colspan=3)
    ax2 = plt.subplot2grid((2, 4), (1, 3), colspan=1)

    # Plot salinity values on ax0
    ax0.plot(sample_nums, psal_vals, '.', color='tab:blue', lw=0.2, alpha=0.6,
             label=f'Bottle file {psal_var}', zorder=2)
    ax0.plot(sample_nums, salinometer_vals, '.', color='tab:orange', lw=0.2, alpha=0.6,
             label='Salinometer', zorder=2)
    ax0.set_xlabel('SAMPLE NUMBER')
    ax0.set_ylabel('Practical salinity')
    ax0.grid(True)

    # Compute correlation coefficient and p-value  (ignoring NaNs)
    valid_corr_mask = ~np.isnan(psal_vals) & ~np.isnan(salinometer_vals)
    if np.count_nonzero(valid_corr_mask) > 1:
        r, pval = pearsonr(psal_vals[valid_corr_mask], salinometer_vals[valid_corr_mask])
        corr_text = f"r = {r:.3f}, p = {pval:.1e}"

    else:
        corr_text = "r = NaN, p = NaN"

    # Annotate correlation on ax0
    ax0.text(0.02, 0.95, corr_text, transform=ax0.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    ax0.legend()

    # Add mplcursors hover annotations to ax0
    cursor = mplcursors.cursor(ax0.collections, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        ind = sel.index
        if ind < len(point_labels):
            sel.annotation.set_text(point_labels[ind])

    # Plot salinity difference on ax1
    ax1.fill_between(sample_num_sorted, sal_diff_sorted, color='k', alpha=0.3,
                     label='Bottle file', lw=0.2, zorder=2)
    ax1.plot(sample_num_sorted, sal_diff_sorted, '.', color='tab:red', alpha=0.8,
             label='Salinometer', lw=0.2, zorder=2)
    ax1.axhline(Sdiff_mean, color='tab:blue', lw=1.6, alpha=0.75, ls=':',
                label=f'Mean = {Sdiff_mean:.2e}')
    ax1.set_xlabel('SAMPLE NUMBER')
    ax1.set_ylabel(f'{psal_var} $-$ salinometer S')
    ax1.grid(True)
    ax1.legend()

    # Plot histogram of salinity difference on ax2
    ax2.hist(sal_diff, bins=20, orientation='horizontal', color='tab:red', alpha=0.7)
    ax2.set_ylim(ax1.get_ylim())
    ax2.axhline(0, color='k', ls='--')
    ax2.axhline(Sdiff_mean, color='tab:blue', lw=1.6, alpha=0.75, ls=':',
                label=f'Mean = {Sdiff_mean:.2e}')
    ax2.set_xlabel('FREQUENCY')
    ax2.set_ylabel(f'{psal_var} $-$ salinometer S')
    ax2.grid(True)
    ax2.legend(loc=0, bbox_to_anchor=(1, 1.2), fontsize = 9)

    fig.suptitle(f'Salinity comparison for samples taken at >{min_pres} dbar (n = {N_count})')
    plt.tight_layout()

    # Button to close the figure
    def close_everything(_):
        plt.close(fig)
        button_exit.close()

    button_exit = widgets.Button(description="Close", layout=widgets.Layout(width='200px'))
    button_exit.on_click(close_everything)
    display(button_exit)