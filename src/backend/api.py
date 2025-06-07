from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
from backend.text_classification import Classifier

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

# Pass the user text to the model and get the classification
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BERT_MODEL_PATH = PROJECT_ROOT / "modell_klassen_notebooks" / "bert_news_classifier"

# To make sure it's the correct path
print(f"Attempting to load model from: {BERT_MODEL_PATH}")

try:
    classifier_instance = Classifier(model_dir=BERT_MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error initializing Classifier: {e}")
    print(f"Please ensure your model is saved at: {BERT_MODEL_PATH}")
    # You might want to exit or disable prediction if the model can't be loaded
    classifier_instance = None # Set to None or handle appropriately

# Define the API endpoint for political leaning prediction
@app.route('/predict_leaning', methods=['POST'])
def predict_leaning():
    """
    Handles POST requests to predict the political leaning of a given text.
    Expects a JSON payload with a 'text' field.
    Returns a JSON object with probabilities for different political leanings.
    """
    # Check if the request body is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Get the JSON data from the request
    data = request.get_json()

    # Extract the 'text' field from the JSON data
    text_to_analyze = data.get('text')

    # Validate if text was provided
    if not text_to_analyze:
        return jsonify({"error": "No 'text' field found in the request"}), 400
    
    # Pass the user text to the model and get the classification
    predicted_label, probabilities = classifier_instance.classify(text_to_analyze, return_probs=True)

    # Log the received text and the simulated response for debugging
    print(f"Received text: '{text_to_analyze}'")
    print(f"Predicted leaning: {probabilities}")

    # Return the JSON response
    return jsonify(probabilities), 200

# This block ensures the Flask app runs only when the script is executed directly
if __name__ == '__main__':
    app.run(debug=True, port=5000)
