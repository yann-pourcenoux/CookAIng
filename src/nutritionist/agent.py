"""Agent for the nutritionist"""

from typing import List, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, Field


# Define the Pydantic model for an ingredient
class Ingredient(BaseModel):
    name: str = Field(description="The name of the ingredient")
    quantity: float = Field(description="The quantity of the ingredient")
    unit: str = Field(description="The unit of measurement for the ingredient")
    quantity_ml: Optional[float] = Field(
        default=None, description="Converted quantity in milliliters"
    )
    quantity_g: Optional[float] = Field(
        default=None, description="Converted quantity in grams"
    )


# Define the Pydantic model for the recipe
class Recipe(BaseModel):
    servings: int = Field(description="The number of servings the recipe makes")
    ingredients: List[Ingredient] = Field(
        description="List of ingredients in the recipe"
    )


class NutritionistAgent:
    def __init__(self, model_name: str = "qwen2.5:14b-instruct-q4_K_M"):
        """
        Initialize the NutritionistAgent.

        Args:
            model_name (str): The name of the Ollama model to use. Defaults to "llama2".
        """
        self.llm = OllamaLLM(model=model_name, temperature=0.0)

    def ask(self, question: str) -> str:
        """
        Ask a question to the nutritionist agent.

        Args:
            question (str): The question to ask the nutritionist agent.

        Returns:
            str: The response from the nutritionist agent.
        """
        return self.llm.predict(question)

    def find_ingredients(self, recipe_text: str) -> Recipe:
        parser = PydanticOutputParser(pydantic_object=Recipe)
        prompt = PromptTemplate(
            template="Extract the ingredients and their quantities from the following recipe:\n\n{recipe}\n\n{format_instructions}\n",
            input_variables=["recipe"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        _input = prompt.format(recipe=recipe_text)
        output = self.llm.invoke(_input)
        return parser.parse(output)

    def convert_units(self, recipe: Recipe) -> Recipe:
        """
        Convert units from cups, tablespoons, and teaspoons to milliliters and grams.

        Args:
            recipe (Recipe): The recipe containing list of ingredients.

        Returns:
            Recipe: The recipe with converted ingredient units.
        """
        unit_conversion_ml = {
            "cup": 240,
            "cups": 240,
            "tablespoon": 15,
            "tablespoons": 15,
            "teaspoon": 5,
            "teaspoons": 5,
        }

        # TODO: Add density_g_per_ml
        density_g_per_ml = {}

        for ingredient in recipe.ingredients:
            unit = ingredient.unit.lower()
            if unit in unit_conversion_ml:
                ml = ingredient.quantity * unit_conversion_ml[unit]
                grams = ml * density_g_per_ml.get(
                    ingredient.name.lower(), 1.0
                )  # Default density
                ingredient.quantity_ml = round(ml, 2)
                ingredient.quantity_g = round(grams, 2)

        return recipe


agent = NutritionistAgent()

ingredients = agent.find_ingredients(
    """
Bien sûr ! Voici une recette saine et équilibrée pour vous aider à prendre du poids de manière saine, tout en incluant votre désir d'avoir du 
riz et du poulet dans le repas.

### Poulet Rôti au Curry et Quinoa au Japonais (avec riz brun)

#### Ingrédients :

- **Pour 4 personnes :**
  - 800g de poitrines de poulet sans peau
  - 1 tasse de quinoa
  - 2 tasses de riz brun cuit (précédemment trempé pendant au moins une heure pour améliorer la digestibilité et la texture)
  - 1 échalote hachée finement
  - 2 gousses d'ail hachées
  - 3 cuillères à soupe de pâte de curry rouge (ou vert, selon vos préférences)
  - 20cl de bouillon de légumes ou de poulet
  - 1 cuillère à café de cumin moulu
  - ½ cuillère à café de cannelle en poudre
  - ¼ cuillère à café de poivre noir concassé
  - Huile d'olive
  - Sel et poivre au goût

#### Pour la garniture :
- Coriandre fraîche hachée (facultatif)
- Concombre en julienne (pour ajouter du croquant)

---

### Préparation :

1. **Préchauffez le four à 200°C**.

2. **Poitrines de poulet :**
   - Assaisonnez les poitrines de poulet avec un peu d'huile d'olive, sel et poivre.
   - Mettez-les dans une assiette à four et laissez-les griller pendant 30 minutes jusqu'à ce qu'elles soient dorées et cuites.

3. **Quinoa :**
   - Faites cuire le quinoa comme indiqué sur l'emballage (1 tasse de quinoa pour 2 tasses d'eau). C'est un grain complet riche en protéines, 
idéal pour la prise de masse tout en restant léger.

4. **Riz brun :**
   - Si vous n'avez pas encore cuit le riz brun à l'avance, faites-le cuire dans une grande quantité d'eau (3 tasses d'eau pour 1 tasse de 
riz) jusqu'à ce qu'il soit tendre et sans eau.

5. **Sauce au curry :**
   - Dans une poêle ou un wok, chauffez quelques gouttes d'huile d'olive.
   - Ajoutez l'échalote hachée et les gousses d'ail et faites revenir jusqu'à ce qu'ils soient tendres (environ 2-3 minutes).
   - Ajoutez la pâte de curry, le cumin moulu, la cannelle et le poivre noir. Remuez pour faire chauffer la sauce.
   - Versez ensuite le bouillon de légumes et laissez mijoter jusqu'à ce que la sauce épaississe (environ 5 minutes).
   
6. **Mélange :**
   - Ajoutez les poitrines de poulet coupées en morceaux dans la sauce curry.
   - Laissez mijoter encore quelques minutes pour que le poulet soit bien imprégné de saveurs.

7. **Montage du plat :**
   - Servez les poitrines de poulet et la sauce sur un lit de riz brun cuit.
   - Ajoutez une cuillère de quinoa au dessus pour varier les textures et ajouter des protéines supplémentaires.
   - Garnissez avec de la coriandre fraîche hachée et du concombre en julienne si vous le souhaitez.

### Conseils :

- **Protéines :** Le poulet est une source excellente de protéines pour aider à construire et à réparer les muscles. La combinaison avec le 
quinoa offre un apport complet en acides aminés.
  
- **Glucides complets :** Le riz brun et le quinoa sont riches en fibres, vitamines, minéraux et glucides complexes qui fournissent une 
énergie stable tout au long de la journée.

Cette recette est parfaitement adaptée pour ceux qui souhaitent prendre du poids de manière saine et contrôlée. Elle fournit un bon équilibre 
entre les protéines, les glucides complets et les fibres.
"""
)

converted_ingredients = agent.convert_units(ingredients)


for ingredient in converted_ingredients.ingredients:
    print(ingredient)
