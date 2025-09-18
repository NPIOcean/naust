'''
Testing naust.salts.py
'''

from pathlib import Path
import pytest
from naust import salts
import numpy as np
import pandas as pd
import xarray as xr
from kval.data import ctd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


# Define test data paths (TT25 data)
@pytest.fixture
def salts_test_data():
    basedir = Path('tests/test_data/salts/tt25')
    return {
        'basedir': basedir,
        'salts_sheet': (basedir / 'salts' /
            'TT25_salinometer_readings_digitized.xlsx'),
        'log_sheet': (
            basedir / 'salts' /
            ('Samplelog_TransectTokt_2025_pelagic'
             '_helium_and_salinity_samples.xlsx')),
        'btl_dir': basedir / 'btl'
    }


def test_read_salts_sheet_parsing(salts_test_data):
    df = salts.read_salts_sheet(salts_test_data['salts_sheet'])

    # Basic structure checks
    assert not df.empty, "Parsed DataFrame is empty"
    assert 'SAL_SAMPLE_NUMBER' in df.columns, "'SAL_SAMPLE_NUMBER' column missing from parsed DataFrame"
    assert pd.api.types.is_integer_dtype(df['SAL_SAMPLE_NUMBER']), "'SAL_SAMPLE_NUMBER' column is not integer type"

    # Check some known values
    assert df.S_median[2] == 34.669
    assert df.SAL_SAMPLE_NUMBER.iloc[-1] == 9999
    assert df.Date[4] == pd.Timestamp('2025-04-03 00:00:00')
    assert list(df.columns) == [
        'Date', 'K15', 'Analyst', 'Salinometer', 'SAL_SAMPLE_NUMBER',
        'S1', 'S2', 'S3', 'S4', 'S5', 'S_median', 'S_offset',
        'S_final', 'Note']

    # Check that 9999 appears only where parsing failed
    assert (df['SAL_SAMPLE_NUMBER'] >= 0).all(), "Negative values found in 'SAL_SAMPLE_NUMBER' column"


def test_salts_sheet_to_csv_export(salts_test_data, tmp_path):
    # Define paths
    xlsx_file = salts_test_data['salts_sheet']
    out_csv = tmp_path / "salinometer_test_output.csv"

    # Run the export
    salts.salts_sheet_to_csv(xlsx_file, out_csv)

    # Check that the file was created
    assert out_csv.exists(), "CSV file was not created"

    # Read back the CSV and do basic checks
    df_csv = pd.read_csv(out_csv)

    assert not df_csv.empty, "Exported CSV is empty"
    assert 'SAL_SAMPLE_NUMBER' in df_csv.columns, "'SAL_SAMPLE_NUMBER' column missing in exported CSV"
    assert df_csv.shape[0] > 5, "Too few rows in exported CSV"


def test_read_salts_log_parsing(salts_test_data):
    df_log = salts.read_salts_log(salts_test_data['log_sheet'])

    # Basic checks
    assert not df_log.empty, "Parsed log DataFrame is empty"

    # Check required columns exist
    expected_cols = [
        'Station', 'CTD LS number', 'Sample name', 'bottle #',
        'Sampling depth (m) from', 'Sample type', 'Sampling date (UTC)',
        'NISKIN_NUMBER', 'intended_sampling_depth', 'SAL_SAMPLE_NUMBER', 'STATION'
    ]
    for col in expected_cols:
        assert col in df_log.columns, f"Column {col} missing in output DataFrame"

    # Check filtering: all Sample type should be 'Salinity'
    assert (df_log['Sample type'] == 'Salinity').all(), "Not all rows are Salinity sample type"

    # Check NISKIN_NUMBER has no nulls and is int
    assert df_log['NISKIN_NUMBER'].notnull().all(), "NISKIN_NUMBER has null values"
    assert pd.api.types.is_integer_dtype(df_log['NISKIN_NUMBER']), "NISKIN_NUMBER is not integer dtype"

    # Check SAL_SAMPLE_NUMBER is integer and extracted correctly (basic)
    assert pd.api.types.is_integer_dtype(df_log['SAL_SAMPLE_NUMBER']), "SAL_SAMPLE_NUMBER is not integer dtype"
    assert (df_log['SAL_SAMPLE_NUMBER'] > 0).any(), "No positive sample numbers found"

    # Check STATION is integer or string of length 3 (zero-padded) 
    STATION_is_int = df_log['STATION'].apply(lambda x: isinstance(x, int)).all()
    STATION_is_3dig_string = df_log['STATION'].apply(lambda x: isinstance(x, str) and len(x) == 3).all()

    assert STATION_is_int or STATION_is_3dig_string, "STATION values not integers or zero-padded strings of length 3"


def merge_salts_sheet_and_log(salts_test_data):
    # Load the cruise log DataFrame using the existing function
    df_log = salts.read_salts_log(salts_test_data['log_sheet'])

    # Load salinometer sheet DataFrame (assumes data is in the first sheet or specify sheet name)
    df_salts = salts.read_salts_sheet(salts_test_data['salts_sheet'])

    # Call the merge function
    ds = salts.merge_salts_sheet_and_log(df_salts, df_log)

    # Check the returned type
    assert isinstance(ds, xr.Dataset), "Output is not an xarray Dataset"

    # Check that dimensions include STATION and NISKIN_NUMBER
    assert 'STATION' in ds.dims, "'STATION' dimension missing in Dataset"
    assert 'NISKIN_NUMBER' in ds.dims, "'NISKIN_NUMBER' dimension missing in Dataset"

    # Check for presence of expected data variables
    expected_vars = {'PSAL_LAB', 'Note', 'intended_sampling_depth', 'Sampling date (UTC)'}
    missing_vars = expected_vars - set(ds.data_vars)
    assert not missing_vars, f"Missing expected variables in Dataset: {missing_vars}"

    # Optionally: check that Dataset is not empty
    assert ds.sizes['STATION'] > 0 and ds.sizes['NISKIN_NUMBER'] > 0, "Dataset dimensions have zero length"

    # Data specific checks
    assert ds.PSAL_LAB.isel(STATION=5, NISKIN_NUMBER=1)==34.676
    assert ds.intended_sampling_depth.isel(STATION=6, NISKIN_NUMBER=1).data == 1000



def test_merge_all_salts_with_btl(salts_test_data):
    # Load salinometer data
    df_salts = salts.read_salts_sheet(salts_test_data['salts_sheet'])
    df_log = salts.read_salts_log(salts_test_data['log_sheet'])

    # Merge salinometer and log data into Dataset
    ds_salts = salts.merge_salts_sheet_and_log(df_salts, df_log)

    # Load BTL Dataset
    ds_btl = ctd.dataset_from_btl_dir(str(salts_test_data['btl_dir']) + '/')

    # Merge all into final Dataset
    ds_combined = salts.merge_all_salts_with_btl(ds_salts, ds_btl)

    # --- Assertions ---
    assert isinstance(ds_combined, xr.Dataset)
    assert 'PSAL_LAB' in ds_combined.data_vars
    assert 'Sdiff1' in ds_combined.data_vars
    assert 'Sdiff2' in ds_combined.data_vars

    # Check dimensions
    assert 'NISKIN_NUMBER_STATION' in ds_combined.dims
    assert not ds_combined['PSAL_LAB'].isnull().any(), "NaNs in S_final after merge"

    # Spot-check difference calculation (example: all diffs should be finite if PSAL1/2 exist)
    if 'PSAL1' in ds_combined and 'PSAL_LAB' in ds_combined:
        assert ds_combined['Sdiff1'].notnull().any(), "Sdiff1 should not be all NaNs"


def test_build_salts_qc_dataset_valid(salts_test_data):
    """
    Test successful creation of the salinity QC dataset from test data.
    """

    log_sheet = salts_test_data['log_sheet']
    salts_sheet = salts_test_data['salts_sheet']
    btl_dir = str(salts_test_data['btl_dir']) + '/'

    ds = salts.build_salts_qc_dataset(log_sheet, salts_sheet, btl_dir)

    assert isinstance(ds, xr.Dataset), "Returned object is not an xarray.Dataset"
    assert 'NISKIN_NUMBER_STATION' in ds.dims, "Missing expected stacked dimension"

    # Expect non-empty dataset
    assert ds.sizes['NISKIN_NUMBER_STATION'] > 0, "Combined dataset is empty"

    # Check key expected variables
    for var in ['PSAL_LAB', 'PSAL1', 'PSAL2', 'Sdiff1', 'Sdiff2']:
        assert var in ds.data_vars, f"Missing expected variable: {var}"
        assert not ds[var].isnull().all(), f"All values of {var} are NaN"

    # Simple physical check: salinity in plausible range
    for sal in ['PSAL_LAB', 'PSAL1', 'PSAL2']:
        if sal in ds:
            assert float(ds[sal].min()) >= 0, f"{sal} has unphysical minimum"
            assert float(ds[sal].max()) <= 45, f"{sal} has unphysical maximum"


def test_build_salts_qc_dataset_missing_file(tmp_path):
    """
    Ensure that missing input files raise appropriate exceptions.
    """

    # Create dummy paths that don't exist
    bad_log = tmp_path / "missing_log.xlsx"
    bad_salts = tmp_path / "missing_salts.xlsx"
    bad_btl_dir = str(tmp_path / "btl" ) + '/'

    with pytest.raises(Exception):
        salts.build_salts_qc_dataset(bad_log, bad_salts, bad_btl_dir)


def test_build_salts_qc_dataset_corrupted_input(salts_test_data, tmp_path):
    """
    Test that corrupted or malformed input raises an error.
    """

    # Create a dummy file that is not a real Excel file
    broken_xlsx = tmp_path / "broken.xlsx"
    broken_xlsx.write_text("Not a real Excel file")

    btl_dir = str(salts_test_data['btl_dir']) + '/'

    # Use real btl_dir but fake sheets
    with pytest.raises(ValueError):
        salts.build_salts_qc_dataset(broken_xlsx, broken_xlsx, btl_dir)


def test_plot_salinity_diff_histogram_runs():
    # Create minimal test Dataset
    n = 50
    pres = np.linspace(0, 1000, n)
    psal = 35 + 0.01 * np.random.randn(n)
    salinometer = 35 + 0.005 * np.random.randn(n)

    ds = xr.Dataset({
        'PSAL1': ('dim_0', psal),
        'PSAL_LAB': ('dim_0', salinometer),
        'PRES': ('dim_0', pres)
    })

    # Run plotting function with defaults
    salts.plot_salinity_diff_histogram(ds, psal_var='PSAL1', salinometer_var='PSAL_LAB', min_pres=200)

    # Check that the current figure is created (optional)
    fig = plt.gcf()
    assert fig is not None
    # Close plot after test to prevent display issues in CI
    plt.close(fig)


def test_plot_salinity_diff_histogram_raises_on_missing_vars():
    ds = xr.Dataset({
        'PSAL1': ('dim_0', np.ones(10)),
        'PRES': ('dim_0', np.ones(10) * 600)
    })
    # Missing salinometer_var 'S_final'
    import pytest
    with pytest.raises(ValueError):
        salts.plot_salinity_diff_histogram(ds, salinometer_var='PSAL_LAB')

    ds2 = xr.Dataset({
        'PSAL_LAB': ('dim_0', np.ones(10)),
        'PRES': ('dim_0', np.ones(10) * 600)
    })
    # Missing psal_var (default)
    with pytest.raises(ValueError):
        salts.plot_salinity_diff_histogram(ds2)

    ds3 = xr.Dataset({
        'PSAL1': ('dim_0', np.ones(10)),
        'PSAL_LAB': ('dim_0', np.ones(10))
        # Missing PRES
    })
    with pytest.raises(ValueError):
        salts.plot_salinity_diff_histogram(ds3)


def test_plot_by_sample_runs_minimal():
    n = 50
    pres = np.linspace(0, 1000, n)
    psal = 35 + 0.01 * np.random.randn(n)
    salinometer = 35 + 0.005 * np.random.randn(n)
    SAL_SAMPLE_NUMBERs = np.arange(n)

    ds = xr.Dataset({
        'PSAL1': ('dim_0', psal),
        'PSAL_LAB': ('dim_0', salinometer),
        'PRES': ('dim_0', pres),
        'SAL_SAMPLE_NUMBER': ('dim_0', SAL_SAMPLE_NUMBERs)
    })

    salts.plot_by_sample(ds)  # Using defaults
    fig = plt.gcf()
    assert fig is not None
    plt.close(fig)



def test_plot_by_sample_raises_on_missing_vars():
    base = np.ones(10)
    sample = np.arange(10)

    # Missing PSAL1
    ds1 = xr.Dataset({
        'PSAL_LAB': ('dim_0', base),
        'PRES': ('dim_0', base * 600),
        'SAL_SAMPLE_NUMBER': ('dim_0', sample)
    })
    with pytest.raises(ValueError, match="PSAL1"):
        salts.plot_by_sample(ds1)

    # Missing PSAL_LAB
    ds2 = xr.Dataset({
        'PSAL1': ('dim_0', base),
        'PRES': ('dim_0', base * 600),
        'SAL_SAMPLE_NUMBER': ('dim_0', sample)
    })
    with pytest.raises(ValueError, match="PSAL_LAB"):
        salts.plot_by_sample(ds2)

    # Missing Sample
    ds3 = xr.Dataset({
        'PSAL1': ('dim_0', base),
        'PSAL_LAB': ('dim_0', base),
        'PRES': ('dim_0', base * 600)
    })
    with pytest.raises(ValueError, match="SAL_SAMPLE_NUMBER"):
        salts.plot_by_sample(ds3)

    # Missing PRES
    ds4 = xr.Dataset({
        'PSAL1': ('dim_0', base),
        'PSAL_LAB': ('dim_0', base),
        'SAL_SAMPLE_NUMBER': ('dim_0', sample)
    })
    with pytest.raises(ValueError, match="PRES"):
        salts.plot_by_sample(ds4)
