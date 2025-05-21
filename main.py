import pandas as pd
import spacy

# Load spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")

# Load dataset
df = pd.read_csv('rec.csv', low_memory=False)

# Preprocess (remove nulls, clean text)
df = df.rename(columns={
    'Name': 'title',
    'RecipeIngredientParts': 'ingredients',
    'RecipeInstructions': 'instructions'
})

df = df[['title', 'ingredients', 'instructions']].dropna()

print(df.head())

# Optionally: Limit dataset for faster demo
# df = df.sample(1000)


def get_similar_recipes(food_query):
    query_doc = nlp(food_query.lower())
    similarities = []

    for idx, row in df.iterrows():
        title_doc = nlp(row['title'].lower())
        sim = query_doc.similarity(title_doc)
        similarities.append((sim, row))

    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:3]

import random

def generate_recipe(food_query):
    similar = get_similar_recipes(food_query)
    ingredients = []
    instructions = []

    for sim, recipe in similar:
        ingredients += recipe['ingredients'].split(", ")
        instructions.append(recipe['instructions'])

    # De-duplicate and shuffle ingredients
    ingredients = list(set(ingredients))
    random.shuffle(ingredients)

    final_instructions = " ".join(instructions[:2])  # Combine top 2

    return {
        "title": f"AI-Generated Recipe: {food_query.title()}",
        "ingredients": ingredients[:10],
        "instructions": final_instructions
    }

if __name__ == "__main__":
    user_input = input("Enter a food name: ")
    recipe = generate_recipe(user_input)

    print("\n" + recipe['title'])
    print("\nIngredients:")
    for ing in recipe['ingredients']:
        print(f"- {ing}")

    print("\nInstructions:")
    print(recipe['instructions'])
