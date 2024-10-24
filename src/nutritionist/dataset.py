"""
This module contains the dataset for the nutritionist agent
"""

import os

import pandas as pd


class Dataset:
    """Class to handle the dataset for the nutritionist agent"""

    df: pd.DataFrame = pd.DataFrame()

    def __init__(self, df: pd.DataFrame):
        self.df = self._init_dataframe(df)
        self.ingredients = self._get_ingredients()

    def _init_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize the dataframe.

        Args:
            df (pd.DataFrame): The dataframe to initialize.

        Returns:
            pd.DataFrame: The initialized dataframe.

        Raises:
            AssertionError: If weight standards are not 100g or if the "Energy (kcal)" column has NaNs.
        """
        # Change the strings in the index to lower case
        df.index = df.index.str.lower()

        # Convert all columns to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.astype(float)

        # Assert that the columns "Energy (kcal)" don't have NaNs
        assert not df["Energy (kcal)"].isna().any(), "Energy (kcal) must not have NaNs"

        # Convert the NaNs to 0s
        df = df.fillna(0)

        assert df["Weight standard (g)"].unique() == [
            100.0
        ], "All weight standards must be 100g"
        return df

    def _get_ingredients(self) -> list[str]:
        """Get the possible ingredients from the dataset."""
        return self.df.index[1:].tolist()

    @classmethod
    def load_from_excel_folder(cls, folder_path: str = "data") -> "Dataset":
        """Load data from a folder of Excel files and return a Dataset instance.

        Args:
            folder_path (str): The path to the folder containing Excel files. Defaults to "data".

        Returns:
            Dataset: An instance of Dataset with data loaded from the Excel files.

        Raises:
            AssertionError: If any file is not an Excel file.
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

            dfs.append(df)

        concatenated_df: pd.DataFrame = pd.concat(dfs, ignore_index=False)

        obj = cls(df=concatenated_df)
        return obj
