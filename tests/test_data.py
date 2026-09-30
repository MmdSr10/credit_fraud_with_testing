import os
from credit_fraud import load_and_inspect_data

def test_data_exists(real_data_path):

    assert os.path.exists(real_data_path)

def test_dataset_non_empty(mock_data_path):
    df = load_and_inspect_data(mock_data_path) 
    assert not df.empty

def test_columns_exist(mock_data_path):

    df = load_and_inspect_data(mock_data_path)

    expected_columns = ["Time", "Amount", "Class"] 

    for col in expected_columns:
        assert col in df.columns

def test_label_values_valid(mock_data_path):
    df=load_and_inspect_data(mock_data_path)
    valid_lobels={0,1}
    unique_labels=set(df["Class"].unique())
    assert unique_labels.issubset(valid_lobels)

def test_shape_correct(mock_data_path):
    df=load_and_inspect_data(mock_data_path)
    assert df.shape[0]>0
    assert df.shape[1]>=3