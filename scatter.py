import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def read_data(filename):
    if filename.endswith('.csv'):
        data = pd.read_csv(filename)
    elif filename.endswith('.xlsx'):
        data = pd.read_excel(filename)
    else:
        raise ValueError("Unsupported file format")
    return data
def analyze_and_visualize(data):
    print("Data Summary:")
    print(data.describe())

   # Get column names
    columns = data.columns

    # Check for at least 3 columns for 3D plot
    if len(columns) >= 3:
        # Choose the first 3 columns for 3D visualization
        x, y, z = columns[:3]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[x], data[y], data[z])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)
        plt.title('3D Scatter Plot')
        plt.show()
    # For 2D plot, choose the first 2 columns
    if len(columns) >= 2:
        x, y = columns[:2]
        plt.figure()
        plt.scatter(data[x], data[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title('2D Scatter Plot')
        plt.show()

def main():
    filename = input("Enter the filename (CSV or Excel): ")
    data = read_data(filename)
    analyze_and_visualize(data)

if __name__ == "__main__":
    main()

