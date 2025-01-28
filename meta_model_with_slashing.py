import json
import os
import requests

# Function to check if model balance file exists and create it if not
def initialize_model_balance():
    model_balance = {
        "model1": {
            "balance": 1000,  # Initial deposit of 1000 euros
        },
        "model2": {
            "balance": 1000, 
        },
        "model3": {
            "balance": 1000, 
        },
        "model3": {
            "balance": 1000, 
        }
    }

    # If the file doesn't exist, create it
    if not os.path.exists("model_balance.json"):
        with open("model_balance.json", "w") as file:
            json.dump(model_balance, file, indent=4)
        print("model_balance.json created with initial values.")
    else:
        # Load existing balance if the file already exists
        with open("model_balance.json", "r") as file:
            model_balance = json.load(file)

    return model_balance

# Function to load model balance data
def load_model_balance():
    with open("model_balance.json", "r") as file:
        return json.load(file)

# Save model balance data
def save_model_balance(model_balance):
    with open("model_balance.json", "w") as file:
        json.dump(model_balance, file, indent=4)

# Function to fetch predictions from each API
def fetch_prediction(api_url, params):
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("prediction")
    except Exception as e:
        print(f"Error fetching from {api_url}: {e}")
        return None

# Function to calculate the weight of each model based on accuracy
def calculate_model_weight(model_name, model_balance):
    accuracy = model_balance.get(model_name, {}).get("accuracy", 0)
    # Higher accuracy models get higher weights, max 1
    return max(0, min(accuracy, 1))


# Map the model predictions (numeric) to class names
label_to_name = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

def consensus_prediction(input_data, model_balance, api_urls):
    predictions = []
    for model_name, api_url in api_urls.items():
        # Call the model API and get the prediction
        response = requests.get(api_url, params=input_data)
        
        # Check if the request was successful
        if response.status_code == 200:
            prediction = response.json().get('prediction', None)
            if prediction is not None:
                # Map the numeric prediction to the class name
                predictions.append(label_to_name.get(prediction, "Unknown"))
            else:
                predictions.append("Error in prediction")
        else:
            predictions.append("Error in API call")

    # For simplicity, we take a majority vote (this can be adjusted)
    if predictions:
        # Return the most common prediction
        return max(set(predictions), key=predictions.count)
    return "No valid predictions"


# Function to handle slashing (penalize models with low accuracy)
def slash_model_balance(model_name, model_balance):
    balance = model_balance.get(model_name, {}).get("balance", 1000)
    accuracy = model_balance.get(model_name, {}).get("accuracy", 0)

    # Slash model balance if accuracy is low (e.g., < 0.6)
    if accuracy < 0.6:
        penalty = balance * 0.2  # Slash 20% of the balance
        new_balance = balance - penalty
        model_balance[model_name]["balance"] = new_balance
        print(f"Model {model_name} slashed! New balance: {new_balance}")
    return model_balance

# Test input (can be changed as needed)
input_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Define API URLs for each model
api_urls = {
    "model1":"https://437c-89-30-29-68.ngrok-free.app/predict", #guilherme
    "model2":"https://6f96-89-30-29-68.ngrok-free.app/predict", #yannis (moi)
    "model3":"https://bb1b-2a04-cec0-11cb-b92b-fea8-9ef7-6b14-9dfb.ngrok-free.app/predict", #dany
    "model4":"https://437c-89-30-29-68.ngrok-free.app/predict"  #théo
}

# Initialize or load model balance data
model_balance = initialize_model_balance()

# Get consensus prediction
result = consensus_prediction(input_data, model_balance, api_urls)
print(f"Consensus Prediction: {result}")

# Apply slashing penalties if accuracy is too low
model_balance = slash_model_balance("model1", model_balance)

# Save updated model balance back to the JSON file
save_model_balance(model_balance)
