'''
Testing naust.salts.py
'''

from pathlib import Path
import pytest
from naust import salts
import pandas as pd
import xarray as xr
from kval.data import ctd

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
    assert 'Sample' in df.columns, "'Sample' column missing from parsed DataFrame"
    assert pd.api.types.is_integer_dtype(df['Sample']), "'Sample' column is not integer type"

    # Check some known values
    assert df.S_median[2] == 34.669
    assert df.Sample.iloc[-1] == 9999
    assert df.Date[4] == pd.Timestamp('2025-04-03 00:00:00')
    assert list(df.columns) == [
        'Date', 'K15', 'Analyst', 'Salinometer', 'Sample',
        'S1', 'S2', 'S3', 'S4', 'S5', 'S_median', 'S_offset',
        'S_final', 'Note']

    # Check that 9999 appears only where parsing failed
    assert (df['Sample'] >= 0).all(), "Negative values found in 'Sample' column"


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
    assert 'Sample' in df_csv.columns, "'Sample' column missing in exported CSV"
    assert df_csv.shape[0] > 5, "Too few rows in exported CSV"


def test_read_salts_log_parsing(salts_test_data):
    df_log = salts.read_salts_log(salts_test_data['log_sheet'])

    # Basic checks
    assert not df_log.empty, "Parsed log DataFrame is empty"

    # Check required columns exist
    expected_cols = [
        'Station', 'CTD LS number', 'Sample name', 'bottle #',
        'Sampling depth (m) from', 'Sample type', 'Sampling date (UTC)',
        'NISKIN_NUMBER', 'intended_sampling_depth', 'sample_number', 'STATION'
    ]
    for col in expected_cols:
        assert col in df_log.columns, f"Column {col} missing in output DataFrame"

    # Check filtering: all Sample type should be 'Salinity'
    assert (df_log['Sample type'] == 'Salinity').all(), "Not all rows are Salinity sample type"

    # Check NISKIN_NUMBER has no nulls and is int
    assert df_log['NISKIN_NUMBER'].notnull().all(), "NISKIN_NUMBER has null values"
    assert pd.api.types.is_integer_dtype(df_log['NISKIN_NUMBER']), "NISKIN_NUMBER is not integer dtype"

    # Check sample_number is integer and extracted correctly (basic)
    assert pd.api.types.is_integer_dtype(df_log['sample_number']), "sample_number is not integer dtype"
    assert (df_log['sample_number'] > 0).any(), "No positive sample numbers found"

    # Check STATION is string of length 3 (zero-padded)
    assert df_log['STATION'].apply(lambda x: isinstance(x, str) and len(x) == 3).all(), "STATION values not zero-padded strings of length 3"


def test_merge_salts_sheet_and_log(salts_test_data):
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
    expected_vars = {'S_final', 'Note', 'intended_sampling_depth', 'Sampling date (UTC)'}
    missing_vars = expected_vars - set(ds.data_vars)
    assert not missing_vars, f"Missing expected variables in Dataset: {missing_vars}"

    # Optionally: check that Dataset is not empty
    assert ds.sizes['STATION'] > 0 and ds.sizes['NISKIN_NUMBER'] > 0, "Dataset dimensions have zero length"

    # Data specific checks
    assert ds.S_final.isel(STATION=5, NISKIN_NUMBER=1)==34.676
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
    assert 'S_final' in ds_combined.data_vars
    assert 'Sdiff1' in ds_combined.data_vars
    assert 'Sdiff2' in ds_combined.data_vars

    # Check dimensions
    assert 'NISKIN_NUMBER_STATION' in ds_combined.dims
    assert not ds_combined['S_final'].isnull().any(), "NaNs in S_final after merge"

    # Spot-check difference calculation (example: all diffs should be finite if PSAL1/2 exist)
    if 'PSAL1' in ds_combined and 'S_final' in ds_combined:
        assert ds_combined['Sdiff1'].notnull().any(), "Sdiff1 should not be all NaNs"