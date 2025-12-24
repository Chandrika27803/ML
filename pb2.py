import pandas as pd
import numpy as np
import argparse
from collections import Counter

 

# Function to load the dataset (CSV or Excel)
def load_dataset(file_path):
    # Determine file type and load accordingly
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    elif file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Only CSV or XLSX allowed.")

    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

 

# Function to analyze column data types and provide suggestions
def analyze_data(df):
    # Get column data types
    column_types = df.dtypes
    print("\n--- Dataset Summary ---")

    num_cols = []
    cat_cols = []

    # Classify columns into numerical or categorical
    for col, dtype in column_types.items():
        if np.issubdtype(dtype, np.number):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    # Suggest analysis techniques based on the column types
    print("\n--- Analysis Recommendations ---")

    if num_cols:
        print(f"Numerical Columns ({len(num_cols)}): {num_cols}")
        print("Recommendation: Use Descriptive Statistics, Correlation Analysis, or Regression Techniques.\n")
        print("Why?")
        print("Numerical data allows you to measure central tendency, variability, and relationships between variables. Descriptive statistics like mean, median, and standard deviation help summarize your data.")
        print("You can also apply regression techniques to model relationships between dependent and independent variables.")

    if cat_cols:
        print(f"Categorical Columns ({len(cat_cols)}): {cat_cols}")
        print("Recommendation: Use Frequency Analysis, Chi-Square Tests, or Decision Trees.\n")
        print("Why?")
        print("Categorical data is often analyzed through frequency counts or distributions. Chi-square tests can be used to determine whether categorical variables are independent. Decision trees can also handle categorical data and are useful for classification tasks.")

    # Special case: if the dataset contains both numerical and categorical columns
    if num_cols and cat_cols:
        print("Mixed Data (Numerical + Categorical)")
        print("Recommendation: Use Statistical Tests (e.g., ANOVA, Chi-Square), Classification Techniques (e.g., Logistic Regression, Decision Trees).\n")
        print("Why?")
        print("When your dataset contains both numerical and categorical data, statistical tests like ANOVA can help you understand whether the means of different groups are significantly different. Classification techniques can model categorical outputs using numerical and categorical inputs.")

    # Further analysis for large datasets
    if df.shape[0] > 10000:
        print(f"Dataset contains {df.shape[0]} rows, which is large.")
        print("Recommendation: Consider sampling the dataset or using Big Data tools (e.g., Apache Spark, Dask) for more efficient analysis.\n")
        print("Why?")
        print("Large datasets can be computationally expensive to analyze. Sampling allows you to work on smaller subsets of the data to speed up the analysis while retaining the characteristics of the full dataset.")

 

# Main function to load and analyze the dataset
def main():
    # Argument parsing for input file
    parser = argparse.ArgumentParser(description="Analyze a dataset and suggest the best techniques.")
    parser.add_argument('file', help="Path to the CSV or Excel file.")
    args = parser.parse_args()

    # Load the dataset
    df = load_dataset(args.file)

    # Analyze the dataset and provide recommendations
    analyze_data(df)

 

if __name__ == "__main__":
    main()

