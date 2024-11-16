# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
