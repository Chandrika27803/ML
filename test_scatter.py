# tests/test_scatter.py
import pandas as pd
import pytest
from scatter import read_data, analyze_and_visualize
import matplotlib.pyplot as plt

def test_read_csv(tmp_path):
    # Arrange: create a temporary CSV file
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b,c\n1,2,3\n4,5,6")

    # Act
    df = read_data(str(csv_file))

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (2, 3)

def test_read_excel(tmp_path):
    # Arrange
    excel_file = tmp_path / "sample.xlsx"
    df_original = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df_original.to_excel(excel_file, index=False)

    # Act
    df = read_data(str(excel_file))

    # Assert
    assert df.equals(df_original)

def test_read_unsupported_format(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("hello")

    with pytest.raises(ValueError):
        read_data(str(bad_file))
        
def test_analyze_and_visualize(monkeypatch, capsys):
    """
    Test that analyze_and_visualize runs without crashing
    AND does not try to open a real window (plt.show mocked).
    """
    # Mock plt.show so plots don't open
    monkeypatch.setattr(plt, "show", lambda: None)

    # Create fake DataFrame
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })

    # Act
    analyze_and_visualize(df)

    # Capture print output
    captured = capsys.readouterr()

    # Assert that summary text printed
    assert "Data Summary:" in captured.out
    assert "count" in captured.out  # printed by df.describe()

