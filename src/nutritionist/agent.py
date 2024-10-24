"""Agent for the nutritionist"""

from typing import List

import pandas as pd
from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser, RetryOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables.config import RunnableConfig
from langchain_ollama import OllamaLLM
from loguru import logger
from pydantic import BaseModel, Field

from nutritionist.dataset import Dataset

load_dotenv()


class Ingredient(BaseModel):
    """Pydantic model for an ingredient."""

    name: str = Field(description="The name of the ingredient")
    quantity: float | None = Field(
        description="The quantity of the ingredient. None if the quantity is not specified"
    )
    unit: str = Field(description="The unit of measurement for the ingredient")


class IngredientGrams(BaseModel):
    """Pydantic model for an ingredient in grams."""

    name: str = Field(description="The name of the ingredient")
    weight: int = Field(description="The weight of the ingredient in grams")


class MatchIngredient(BaseModel):
    """Pydantic model for a matching ingredients to the dataset."""

    found_match: bool = Field(
        description="Whether the ingredient was found in the list of possible ingredients"
    )
    closest_match: str = Field(
        description="The closest match found in the list of possible ingredients"
    )


class Recipe(BaseModel):
    """Pydantic model for a recipe read from the input."""

    servings: int = Field(description="The number of servings the recipe makes")
    ingredients: List[Ingredient] = Field(
        description="List of ingredients in the recipe"
    )


class RecipeGrams(BaseModel):
    """Pydantic model for a recipe with ingredients in grams."""

    servings: int = Field(description="The number of servings the recipe makes")
    ingredients: List[IngredientGrams] = Field(
        description="List of ingredients in the recipe"
    )


class NutritionistAgent:
    """Nutritionist AI Agent."""

    def __init__(
        self, model_name: str = "qwen2.5:14b-instruct-q4_K_M", use_retry: bool = True
    ):
        """
        Initialize the NutritionistAgent.

        Args:
            model_name (str): The name of the Ollama model to use. Defaults to "qwen2.5:14b-instruct-q4_K_M".
            use_retry (bool): Whether to use retry parsing. Defaults to True.
        """
        self.llm = OllamaLLM(model=model_name, temperature=0.0)
        self.dataset = Dataset.load_from_excel_folder()
        self.use_retry = use_retry

    def _clean_recipe_text(self, recipe_text: str) -> str:
        """Clean the recipe text.

        Args:
            recipe_text (str): The text of the recipe.

        Returns:
            str: The cleaned recipe text.
        """
        recipe_text = recipe_text.encode("utf-8").decode("utf-8")
        return recipe_text

    def _extract_ingredients(self, recipe_text: str) -> Recipe:
        """
        Find the ingredients in the recipe.

        Args:
            recipe_text (str): The text of the recipe.

        Returns:
            Recipe: The recipe with the ingredients.
        """
        parser = PydanticOutputParser(pydantic_object=Recipe)
        prompt = PromptTemplate(
            template="Extract the ingredients and their quantities from the following recipe:\n\n{recipe}\n\n{format_instructions}\n",
            input_variables=["recipe"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        _input = prompt.format_prompt(recipe=recipe_text)
        output = self.llm.invoke(
            _input, config=RunnableConfig(tags=["extract_ingredients"])
        )
        return parser.parse(output)

    def _get_grams_converter_parser(self) -> PydanticOutputParser | RetryOutputParser:
        """Get the parser for converting ingredients to grams."""
        parser = PydanticOutputParser(pydantic_object=IngredientGrams)
        if self.use_retry:
            return RetryOutputParser.from_llm(parser=parser, llm=self.llm)
        return parser

    def _get_grams_converter_prompt(
        self, ingredient: Ingredient, parser_instructions: str
    ) -> PromptValue:
        """Get the prompt for converting an ingredient to grams.

        Args:
            ingredient (Ingredient): The ingredient to convert.
            parser_instructions (str): The instructions for the parser.

        Returns:
            PromptTemplate: The prompt for converting an ingredient to grams.
        """
        template = "Convert the quantity of the following ingredient to grams.\n"

        if ingredient.unit:
            template_method = "To do so, you must first estimate the weight in grams of one {ingredient.unit} and then multiply it by the number of {ingredient.unit}."
            string_with_unit = f"{ingredient.unit} of "
        else:
            template_method = "To do so, you must first estimate the weight in grams of one {ingredient.name} and then multiply it by the number of {ingredient.name}."
            string_with_unit = ""

        template = (
            template
            + f"INGREDIENT:\n{ingredient.quantity} {string_with_unit}{ingredient.name}\n\n"
        )
        template = template + template_method
        template = template + "\n\n{format_instructions}"

        prompt_template = PromptTemplate(
            template=(template),
            input_variables=["ingredient"],
            partial_variables={"format_instructions": parser_instructions},
        )

        prompt = prompt_template.format_prompt(ingredient=ingredient)

        return prompt

    def _convert_to_grams(self, recipe: Recipe) -> RecipeGrams:
        """
        Convert units from cups, tablespoons, and teaspoons to grams.

        Args:
            recipe (Recipe): The recipe containing list of ingredients.

        Returns:
            RecipeGrams: The recipe with converted ingredient units.
        """
        parser = self._get_grams_converter_parser()

        ingredients_grams: List[IngredientGrams] = []
        for ingredient in recipe.ingredients:
            if ingredient.quantity is None:
                recipe.ingredients.remove(ingredient)
                continue

            prompt = self._get_grams_converter_prompt(
                ingredient, parser.get_format_instructions()
            )
            output = self.llm.invoke(
                prompt,
                config=RunnableConfig(tags=["convert_to_grams", ingredient.name]),
            )

            if self.use_retry:
                try:
                    ingredient_grams = parser.parse_with_prompt(output, prompt)
                except ValueError as e:
                    logger.error(f"Failed to convert ingredient {ingredient.name}: {e}")
            else:
                ingredient_grams = parser.parse(output)

            ingredients_grams.append(ingredient_grams)

        converted_recipe = RecipeGrams(
            servings=recipe.servings, ingredients=ingredients_grams
        )
        return converted_recipe

    def _rename_ingredients(self, recipe: RecipeGrams) -> RecipeGrams:
        """
        Rename the ingredients to the closest available names in the dataset using a language model.

        Args:
            recipe (RecipeGrams): The recipe containing the list of ingredients.

        Returns:
            RecipeGrams: The recipe with renamed ingredient names.
        """
        possible_ingredients_string = "\n".join(
            ["- " + ingredient for ingredient in self.dataset.ingredients]
        )

        parser = PydanticOutputParser(pydantic_object=MatchIngredient)
        prompt = PromptTemplate(
            template="Find the closest matching ingredient for '{ingredient.name}' from the possible ingredients:\n{possible_ingredients}\n\n{format_instructions}.",
            input_variables=["ingredient"],
            partial_variables={
                "possible_ingredients": possible_ingredients_string,
                "format_instructions": parser.get_format_instructions(),
            },
        )

        out_recipe = RecipeGrams(servings=recipe.servings, ingredients=[])
        for ingredient in recipe.ingredients:
            input = prompt.format_prompt(ingredient=ingredient)
            output = self.llm.invoke(
                input,
                config=RunnableConfig(tags=["rename_ingredient", ingredient.name]),
            )

            output = parser.parse(output)
            if output.found_match:
                assert (
                    output.closest_match in self.dataset.ingredients
                ), "The closest match must be in the list of possible ingredients"
                out_recipe.ingredients.append(
                    IngredientGrams(name=output.closest_match, weight=ingredient.weight)
                )
            else:
                logger.warning(f"No match found for ingredient: {ingredient.name}")

        return out_recipe

    def _compute_calories(self, recipe: RecipeGrams) -> pd.Series:
        """
        Compute the calories for all ingredients in the recipe.

        Args:
            recipe (RecipeGrams): The recipe containing a list of ingredients.

        Returns:
            pd.Series: The recipe with calculated calories for each ingredient.
        """
        total_calories: pd.Series | None = None
        for ingredient in recipe.ingredients:
            calories = self.dataset.df.loc[ingredient.name]
            calories = calories * ingredient.weight / calories["Weight standard (g)"]

            if total_calories is None:
                total_calories = calories
            else:
                total_calories += calories

        total_calories = total_calories.round(0)
        total_calories = total_calories.astype(int)
        total_calories.name = "Recipe"

        return total_calories

    def _compute_calories_per_serving(
        self, recipe: RecipeGrams, total_calories: pd.Series
    ) -> pd.Series:
        """Compute the calories per serving for the recipe.

        Args:
            recipe (RecipeGrams): The recipe containing a list of ingredients.
            total_calories (pd.Series): The total calories for the recipe.

        Returns:
            pd.Series: The calories per serving for the recipe.
        """
        return total_calories / recipe.servings

    def analyze_recipe(self, recipe_text: str, per_serving: bool = True) -> pd.Series:
        """Analyze a recipe.

        Args:
            recipe_text (str): The text of the recipe.
            per_serving (bool): Whether to return the calories per serving. Defaults to True.

        Returns:
            pd.Series: The calories for the recipe.
        """
        recipe_text = self._clean_recipe_text(recipe_text)
        recipe = self._extract_ingredients(recipe_text)
        recipe_grams = self._convert_to_grams(recipe)
        recipe_grams = self._rename_ingredients(recipe_grams)
        total_calories = self._compute_calories(recipe_grams)
        if per_serving:
            return self._compute_calories_per_serving(recipe_grams, total_calories)
        return total_calories


def main():
    # Load the recipe from a txt file
    with open("recipe_chicken.txt", "r") as file:
        recipe_text = file.read()

    agent = NutritionistAgent()
    calories = agent.analyze_recipe(recipe_text, per_serving=True)
    print(calories)
