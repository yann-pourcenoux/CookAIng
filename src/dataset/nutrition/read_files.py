import os

import pandas as pd


def _process_excel_file(file_path: str) -> pd.DataFrame:
    """
    Process an Excel (.xlsx) file and return a DataFrame.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: DataFrame containing data from the Excel file.
    """
    df = pd.read_excel(file_path)
    # Keep only the first two columns
    df = df.iloc[:, :2]

    # Transpose the dataframe
    df = df.T
    df.reset_index(inplace=True)
    # Rename the columns
    df.columns = df.iloc[0]
    df.drop(df.index[0], inplace=True)
    return df


def read_excel_files(folder_path: str) -> pd.DataFrame:
    """
    Read all Excel (.xlsx) files from a folder and aggregate them into one DataFrame.

    Args:
        folder_path (str): Path to the folder containing Excel files.

    Returns:
        pd.DataFrame: Aggregated DataFrame containing data from all Excel files.

    Raises:
        FileNotFoundError: If the specified folder does not exist.
        ValueError: If no Excel files are found in the specified folder.
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder '{folder_path}' does not exist.")

    # List all Excel files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(".xlsx")]

    # Check if any Excel files were found
    if not excel_files:
        raise ValueError(f"No Excel files found in the folder '{folder_path}'.")

    # Read and concatenate all Excel files
    dfs = []
    for file in excel_files:
        file_path = os.path.join(folder_path, file)
        df = _process_excel_file(file_path)
        dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    # Change the index to be the Nutrient column
    combined_df.set_index("Nutrient (unit)", inplace=True)

    return combined_df


# Example usage
folder_path = "/home/yann/CookAIng/data/raw"
result_df = read_excel_files(folder_path)
print(result_df)
