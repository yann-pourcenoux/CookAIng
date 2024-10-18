"""
This module contains the dataset for the nutritionist agent
"""

import os

import pandas as pd


class Dataset:
    """Class to handle the dataset for the nutritionist agent"""

    df: pd.DataFrame = pd.DataFrame()

    @classmethod
    def load_from_csv(cls, csv_path: str) -> "Dataset":
        """Load data from a CSV file and return a Dataset instance.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            Dataset: An instance of Dataset with data loaded from the CSV.
        """
        dataset = cls(
            folder_path=csv_path.rsplit("/", 1)[0] if "/" in csv_path else "."
        )
        dataset.df = pd.read_csv(csv_path)
        return dataset

    def to_csv(self, csv_path: str) -> None:
        """Convert the dataset to a CSV file.

        Args:
            csv_path (str): The path to save the CSV file.
        """
        self.df.to_csv(csv_path, index=False)

    @classmethod
    def load_from_excel_folder(cls, folder_path: str) -> "Dataset":
        """Load data from a folder of Excel files and return a Dataset instance.

        Args:
            folder_path (str): The path to the folder containing Excel files.

        Returns:
            Dataset: An instance of Dataset with data loaded from the Excel files.
        """
        dfs: list[pd.DataFrame] = []
        for file in os.listdir(folder_path):
            file_path: str = os.path.join(folder_path, file)
            assert file_path.endswith(".xlsx"), "All files must be Excel files"

            df: pd.DataFrame = pd.read_excel(file_path)
            # Keep only the first column as a DataFrame
            df = df.iloc[:, [0, 1]]
            # Set the index to be the first row (original first column)
            df.index = df.iloc[:, 0]
            # Drop the first column
            df = df.drop(df.columns[0], axis=1)
            # Transpose the DataFrame
            df = df.transpose()

            # Drop the first row after setting it as the index
            # df = df.drop(df.index[0], axis=0)
            # print(df.head())
            # Append the processed DataFrame to the list
            dfs.append(df)

        cls.df = pd.concat(dfs, ignore_index=False)
        print(cls.df.head())
        return cls


Dataset.load_from_excel_folder("data")
