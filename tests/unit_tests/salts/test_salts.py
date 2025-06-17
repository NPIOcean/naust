'''
Testing naust.salts.py
'''

from pathlib import Path
import pytest
from naust import salts
import pandas as pd

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

