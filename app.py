from flask import Flask, request, jsonify
import joblib  # For loading saved models

# Load your trained model
model = joblib.load("svm_model.pkl")

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Parse input arguments from the query string
        sepal_length = float(request.args.get('sepal_length'))
        sepal_width = float(request.args.get('sepal_width'))
        petal_length = float(request.args.get('petal_length'))
        petal_width = float(request.args.get('petal_width'))

        # Perform prediction
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        # Map numeric prediction to class label
        class_labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_class = class_labels[int(prediction[0])]

        # Response structure
        response = {
            "input": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width
            },
            "prediction": predicted_class,
            "success": True
        }
    except Exception as e:
        # Handle errors
        response = {"error": str(e), "success": False}

    return jsonify(response)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)