import pandas as pd
import numpy as np
import pytest
import os
from pb2 import load_dataset, analyze_data, main

def test_load_datsetcsv(tmp_path):
    # Arrange: create a temporary CSV file
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

    # Act
    df = load_dataset(str(csv_file))

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)

def test_load_datset(tmp_path):
    # Arrange
    excel_file = tmp_path / "sample.xlsx"
    df_original = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df_original.to_excel(excel_file, index=False)

    # Act
    df = load_dataset(str(excel_file))

    # Assert
    assert df.equals(df_original)
    
def test_read_unsupported_format(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("hello")

    with pytest.raises(ValueError):
        load_dataset(str(bad_file))
        
def test_analyze_data(capsys):
    df=pd.DataFrame({
        "a":[1,2,3],
        "b":[4,5,6],
        "c":["apple","bat","cat"],
        "d":["dog","donkey","monkey"]
        })
    file_path="/home/student/Desktop/python/mltest.py/Bird_strikes.csv"
    df1=load_dataset(file_path)
    analyze_data(df1)
    captured=capsys.readouterr()
    assert "\n--- Dataset Summary ---" in captured.out
    assert "\n--- Analysis Recommendations ---" in captured.out
    assert "Recommendation: Use Descriptive Statistics, Correlation Analysis, or Regression Techniques." in captured.out
    assert "Recommendation: Use Frequency Analysis, Chi-Square Tests, or Decision Trees." in captured.out
    assert "Recommendation: Use Statistical Tests (e.g., ANOVA, Chi-Square), Classification Techniques (e.g., Logistic Regression, Decision Trees)." in captured.out
    assert "Recommendation: Consider sampling the dataset or using Big Data tools (e.g., Apache Spark, Dask) for more efficient analysis." in captured.out
    


