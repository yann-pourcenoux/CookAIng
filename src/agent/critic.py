import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, Tool
from langchain.chat_models import ChatOllama
from langchain.tools import BaseTool


def split_ingredients(recipe: str) -> List[str]:
    """
    Splits the recipe string into individual ingredient lines.

    Args:
        recipe (str): The recipe as a multiline string.

    Returns:
        List[str]: A list of ingredient lines.
    """
    return [line.strip() for line in recipe.strip().split("\n") if line.strip()]


def parse_ingredient(ingredient_line: str) -> Tuple[float, str]:
    """
    Parses an ingredient line to extract the quantity and ingredient name.

    Args:
        ingredient_line (str): A single line from the recipe.

    Returns:
        Tuple[float, str]: A tuple containing the quantity and the ingredient name.
    """
    pattern = r"(?P<quantity>\d+(?:\.\d+)?)g\s+of\s+(?P<ingredient>.+)"
    match = re.match(pattern, ingredient_line, re.IGNORECASE)
    if match:
        quantity = float(match.group("quantity"))
        ingredient = match.group("ingredient").strip().lower()
        return quantity, ingredient
    else:
        raise ValueError(f"Unable to parse ingredient line: {ingredient_line}")


def load_nutrition_data(file_path: str) -> pd.DataFrame:
    """
    Loads the nutrition data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the nutrition CSV file.

    Returns:
        pd.DataFrame: The loaded nutrition data.
    """
    return pd.read_csv(file_path)


def get_nutrition(ingredient: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Retrieves the nutrition information for a given ingredient.

    Args:
        ingredient (str): The name of the ingredient.
        df (pd.DataFrame): The nutrition DataFrame.

    Returns:
        Optional[Dict[str, Any]]: A dictionary of nutrition information if found, else None.
    """
    row = df[df["ingredient"].str.lower() == ingredient.lower()]
    if not row.empty:
        return row.iloc[0].to_dict()
    return None


def compute_nutrition(
    quantity: float, nutrition_info: Dict[str, Any]
) -> Dict[str, float]:
    """
    Computes the nutrition values for a given quantity of an ingredient.

    Args:
        quantity (float): The quantity in grams.
        nutrition_info (Dict[str, Any]): The nutrition information per 100g.

    Returns:
        Dict[str, float]: The computed nutrition values for the given quantity.
    """
    return {
        k: (v / 100) * quantity for k, v in nutrition_info.items() if k != "ingredient"
    }


def aggregate_nutrition(nutrition_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates a list of nutrition dictionaries into a total nutrition dictionary.

    Args:
        nutrition_list (List[Dict[str, float]]): A list of nutrition dictionaries.

    Returns:
        Dict[str, float]: The aggregated total nutrition.
    """
    total: Dict[str, float] = {}
    for nutrition in nutrition_list:
        for key, value in nutrition.items():
            total[key] = total.get(key, 0) + value
    return total


class NutritionCalculatorTool(BaseTool):
    """
    A tool to calculate the total nutrition of a recipe.
    """

    name = "NutritionCalculator"
    description = "Calculates the nutritional values of a dish based on the recipe and nutrition data."

    def __init__(self, nutrition_df: pd.DataFrame):
        """
        Initializes the NutritionCalculatorTool with the given nutrition DataFrame.

        Args:
            nutrition_df (pd.DataFrame): The nutrition DataFrame.
        """
        self.nutrition_df = nutrition_df

    def _run(self, query: str) -> Dict[str, Any]:
        """
        Runs the nutrition calculation.

        Args:
            query (str): The recipe string.

        Returns:
            Dict[str, Any]: The total nutrition of the dish.
        """
        ingredients = split_ingredients(query)
        nutrition_list = []
        for item in ingredients:
            quantity, ingredient = parse_ingredient(item)
            nutrition_info = get_nutrition(ingredient, self.nutrition_df)
            if nutrition_info:
                nutrition = compute_nutrition(quantity, nutrition_info)
                nutrition_list.append(nutrition)
            else:
                raise ValueError(f"Nutrition information for '{ingredient}' not found.")
        total_nutrition = aggregate_nutrition(nutrition_list)
        return total_nutrition

    def _arun(self, query: str) -> Dict[str, Any]:
        """
        Asynchronous run method. Not implemented.

        Args:
            query (str): The recipe string.

        Raises:
            NotImplementedError: Always, since async is not supported.
        """
        raise NotImplementedError("NutritionCalculatorTool does not support async")


def build_agent(nutrition_csv_path: str) -> AgentExecutor:
    """
    Builds the LangChain agent for nutrition calculation.

    Args:
        nutrition_csv_path (str): Path to the nutrition CSV file.

    Returns:
        AgentExecutor: The configured LangChain agent.
    """
    nutrition_df = load_nutrition_data(nutrition_csv_path)
    nutrition_tool = NutritionCalculatorTool(nutrition_df=nutrition_df)
    tools = [Tool.from_tool(nutrition_tool)]

    llm = ChatOllama(model="gemma2:9b-instruct-q4_0")

    agent = AgentExecutor.from_agent_and_tools(agent=llm, tools=tools, verbose=True)

    return agent


def main() -> None:
    """
    Main function to execute the Nutrition Agent.

    Parses the recipe, computes the nutrition, and prints the results.
    """
    recipe = """
    150g of salmon oven baked
    150g of green beans stir fried
    50g of peas stir fried
    """

    nutrition_csv = "path/to/nutrition_data.csv"
    agent = build_agent(nutrition_csv_path=nutrition_csv)

    try:
        total_nutrition = agent.run(recipe)
        print("Total Nutrition for the Dish:")
        for key, value in total_nutrition.items():
            print(f"{key}: {value:.2f}")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    load_dotenv()  # Load environment variables from a .env file if present
    main()
