import os
from QualityClassifierAbstracter import classify_image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import load_model


from dietplanner import DietTypeRecommender

app = Flask(__name__)

# Specify folder to save uploaded images
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Route to render the main page (index.html)
@app.route('/')
def index():
    return render_template('index.html')


# Route to render diet_planner.html
@app.route('/diet-planner')
def diet_planner():
    return render_template('diet_planner.html')


# Route to render dish_classifier.html
@app.route('/dish-classifier')
def dish_classifier():
    return render_template('dish_classifier.html')


# Route to render quality_classifier.html
@app.route('/quality-classifier')
def quality_classifier():
    return render_template('quality_classifier.html')


# Endpoint to process diet recommendations
@app.route('/api/recommend-diet', methods=['POST'])
def recommend_diet():
    # Retrieve form data
    height = request.form.get('height')
    weight = request.form.get('weight')
    preference = request.form.get('diet-preference')
    goal = request.form.get('goal')
    # age = request.form.get('age')

    # Initialize the DietTypeRecommender
    recommender = DietTypeRecommender()

    # Call the getMeal method to get diet recommendations
    recommendations = recommender.getMeal(int(height), int(weight), preference, goal)

    # Return recommendations as JSON
    # Prepare the response in the expected format
    response = {
        "recommendations": [
            {
                "Diet_type": recommendations["predicted_recipe"]["Diet_type"],
                "Recipe_name": recommendations["predicted_recipe"]["Recipe_name"],
                "Protein(g)": recommendations["predicted_recipe"]["Protein(g)"],
                "Carbs(g)": recommendations["predicted_recipe"]["Carbs(g)"],
                "Fat(g)": recommendations["predicted_recipe"]["Fat(g)"],
            }
        ]
    }

    # Return recommendations as JSON
    return jsonify(response)

# Endpoint to process diet recommendations
@app.route('/api/recommend-more-diet', methods=['POST'])
def recommend_more_diet():
    # Retrieve form data
    Diet_type = request.form.get('Diet_type')
    Recipe_name = request.form.get('Recipe_name')

    # Initialize the DietTypeRecommender
    recommender = DietTypeRecommender()

    # Call the getMeal method to get diet recommendations
    recommendations = recommender.getMoreMoreMealRecommendation(Diet_type,Recipe_name)

    # Return recommendations as JSON
    # Prepare the response in the expected format
    response = {
        "recommendations": recommendations
    }

    # Return recommendations as JSON
    return jsonify(response)


@app.route('/api/classify-quality', methods=['POST'])
def classify_quality():
    file = request.files['quality-image']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    img_height = 180
    img_width = 180
    loaded_model = load_model("QualityClassifierTrainedModel")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    if file and allowed_file(file.filename):
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)

        # Save the file
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_path)
    # Call the classify_image function from script.py
        response = classify_image(full_path, base_model, loaded_model)

        # Return the result as JSON (for simplicity, assuming response contains the class prediction)
        return jsonify({"quality": response}), 200

import requests

url = 'https://41e2-35-190-155-189.ngrok-free.app/call_function'  # Replace with your ngrok public URL



@app.route('/api/predict-dish', methods=['POST'])
def dish_classify():
    file = request.files['dish-image']

    # If no file is selected
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        # Secure the filename to prevent security issues
        filename = secure_filename(file.filename)

        # Save the file
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_path)
    # Call the classify_image function from script.py
        response = requests.post(url)

        # Return the result as JSON (for simplicity, assuming response contains the class prediction)
        return jsonify({"prediction": response.text}), 200
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
