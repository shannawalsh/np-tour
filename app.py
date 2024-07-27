from flask import Flask, render_template, request, jsonify
import logging
from datetime import datetime

# app will run at: http://127.0.0.1:5000/

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")

# Initialize the Flask application
app = Flask(__name__)

# Define the route for the home page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")
  
# Define the route for the plan trip page
@app.route("/plan_trip", methods=["GET"])
def plan_trip():
  return render_template("plan-trip.html")

# Define the route for view trip page with the generated trip itinerary
@app.route("/view_trip", methods=["POST"])
def view_trip():
  return render_template("view-trip.html")
    
# Run the flask server
if __name__ == "__main__":
    app.run()
