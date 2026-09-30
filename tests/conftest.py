import pytest
import pandas as pd
import os

@pytest.fixture
def mock_data_path(tmp_path):
    
    df = pd.DataFrame({
        "Time": [0.0, 1.0, 2.0],
        "Amount": [10.5, 20.0, 50.2],
        "V1": [0.1, -0.2, 0.3],
        "Class": [0, 1, 0]
    })

    file_path=tmp_path/'mock_creditcard.csv'

    df.to_csv(file_path,index=False)
    
    return str(file_path)

@pytest.fixture
def real_data_path():
    
    return "creditcard.csv"