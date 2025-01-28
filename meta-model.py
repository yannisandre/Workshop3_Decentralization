import requests

# List of group members' API URLs (replace with actual ngrok URLs)
api_urls = [
    "https://437c-89-30-29-68.ngrok-free.app/predict", #guilherme
    "https://6f96-89-30-29-68.ngrok-free.app/predict", #yannis (moi)
    "https://33a5-89-30-29-68.ngrok-free.app/predict", #dany
    "https://ee5d-89-30-29-68.ngrok-free.app/predict"  #théo
]

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

# Consensus prediction logic
def consensus_prediction(input_data):
    predictions = []
    for url in api_urls:
        prediction = fetch_prediction(url, input_data)
        if prediction is not None:
            predictions.append(prediction)

    # Normalize the class labels to lowercase to avoid case-related issues
    predictions = [pred.lower() if isinstance(pred, str) else pred for pred in predictions]

    # Convert class labels to numeric (setosa = 0, versicolor = 1, virginica = 2)
    label_to_num = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    num_to_label = {v: k for k, v in label_to_num.items()}

    # Ensure all predictions are valid labels
    valid_predictions = [pred for pred in predictions if pred in label_to_num]

    if not valid_predictions:
        return "Error: No valid predictions received."

    numeric_predictions = [label_to_num[pred] for pred in valid_predictions]

    # Calculate average and return the nearest class
    average_prediction = round(sum(numeric_predictions) / len(numeric_predictions))
    return num_to_label[average_prediction]

# Test input (can be changed as needed)
input_data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Get consensus prediction
result = consensus_prediction(input_data)
print(f"Consensus Prediction: {result}")
