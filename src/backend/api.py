from flask import Flask, request, jsonify
from flask_cors import CORS
<<<<<<< HEAD
import random

# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This is crucial because your HTML frontend is likely running on a different origin
# (e.g., a local file, or a different port) than this Flask API.
# Without CORS, your browser would block the API requests for security reasons.
CORS(app)

=======
from pathlib import Path
from backend.text_classification import Classifier

# Initialize the Flask application
app = Flask(__name__)
CORS(app)

try:
    classifier_instance = Classifier()
except FileNotFoundError as e:
    print(f"Error initializing Classifier: {e}")
    classifier_instance = None # Set to None or handle appropriately

>>>>>>> origin/main
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
<<<<<<< HEAD

    # --- Dummy Political Leaning Prediction Logic ---
    # In a real application, you would integrate your machine learning model here.
    # For this example, we'll generate random probabilities that sum to 1.0.

    # Generate 5 random numbers
    r1 = random.random()
    r2 = random.random()
    r3 = random.random()
    r4 = random.random()
    r5 = random.random()

    # Sum them up
    total = r1 + r2 + r3 + r4 + r5

    # Normalize them so they sum to 1.0
    # Ensure no division by zero, though random.random() makes this unlikely
    if total == 0:
        # If all are zero, assign equal probabilities to avoid division by zero
        left_prob = left_center_prob = center_prob = right_center_prob = right_prob = 0.2
    else:
        left_prob = r1 / total
        left_center_prob = r2 / total
        center_prob = r3 / total
        right_center_prob = r4 / total
        right_prob = r5 / total

    # Create the response dictionary
    response_data = {
        "left": left_prob,
        "left-center": left_center_prob,
        "center": center_prob,
        "right-center": right_center_prob,
        "right": right_prob
    }

    # Log the received text and the simulated response for debugging
    print(f"Received text: '{text_to_analyze}'")
    print(f"Simulated leaning: {response_data}")

    # Return the JSON response
    return jsonify(response_data), 200

# This block ensures the Flask app runs only when the script is executed directly
if __name__ == '__main__':
    # Run the Flask app on localhost, port 5000.
    # debug=True allows for automatic reloading on code changes and provides a debugger.
=======
    
    # Pass the user text to the model and get the classification
    predicted_label, probabilities = classifier_instance.classify(text_to_analyze, return_probs=True)

    # Log the received text and the simulated response for debugging
    print(f"Received text: '{text_to_analyze}'")
    print(f"Predicted leaning: {probabilities}")

    # Return the JSON response
    return jsonify(probabilities), 200

# This block ensures the Flask app runs only when the script is executed directly
if __name__ == '__main__':
>>>>>>> origin/main
    app.run(debug=True, port=5000)
