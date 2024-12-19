import requests

url = "http://127.0.0.1:5000/predict"

# Exemple 1 de pingüí
penguin_sample_1 = {
    "island": "Torgersen",
    "bill_length_mm": 39.1,
    "bill_depth_mm": 18.7,
    "flipper_length_mm": 181,
    "body_mass_g": 3750,
    "sex": "Male"
}

# Exemple 2 de pingüí
penguin_sample_2 = {
    "island": "Dream",
    "bill_length_mm": 40.2,
    "bill_depth_mm": 17.4,
    "flipper_length_mm": 190,
    "body_mass_g": 3800,
    "sex": "Female"
}

# Llista de models
models = ["logistic_regression", "svm", "decision_tree", "knn"]

# Realitzar dues peticions per model
for model_name in models:
    for idx, sample in enumerate([penguin_sample_1, penguin_sample_2], start=1):
        response = requests.post(f"{url}/{model_name}", json=sample)
        print(f"Model: {model_name}, Petició {idx}")
        print(response.json())
