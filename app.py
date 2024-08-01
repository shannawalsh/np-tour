from flask import Flask, render_template, request, jsonify
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
import logging
from datetime import datetime

# Initialize the OpenAI language model
llm = OpenAI()

# app will run at: http://127.0.0.1:5000/

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")

# Initialize the Flask application
app = Flask(__name__)

# Define the function to build the new trip prompt
def build_new_trip_prompt(form_data):
  examples = [
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Yosemite National Park. They are traveling with one partner, in a group. They leave on 2024-08-18 and return home on 2024-08-20. They prefer to stay in campsites. They enjoy hiking, swimming. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yosemite Trip 2024","location":"Yosemite National Park","trip_start":"2024-08-18","trip_end":"2024-08-20","num_days":"3","traveling_with":"solo, with kids","lodging":"campsites","adventure":"hiking, swimming","itinerary":[{{"Day 1":"Arrival and Camp Setup (2024-08-18)","Morning":"Arrive at Yosemite National Park and check into your reserved campsite at Upper Pines Campground.","Afternoon":"Set up your campsite and explore the area. Take a short hike to Mirror Lake to acclimate to the surroundings.","Dinner":"Dinner at the campsite with a view of Half Dome. Prepare a meal using your camp stove or grill."}},{{"Day 2":"Exploring Yosemite Valley (2024-08-19)","Morning":"Start the day with a hike on the Mist Trail to Vernal Fall and Nevada Fall. This is a moderately challenging hike with spectacular waterfall views.","Lunch":"Picnic lunch by the Merced River. Pack sandwiches, snacks, and drinks.","Afternoon":"Visit the Yosemite Valley Visitor Center to learn about the park's history and geology.","Dinner":"Dine at the Yosemite Valley Lodge's Base Camp Eatery for a variety of casual dining options."}},{{"Day 3":"Glacier Point Adventure (2024-08-20)","Morning":"Drive up to Glacier Point for breathtaking panoramic views of Yosemite Valley, Half Dome, and the High Sierra.","Lunch":"Enjoy lunch at the Glacier Point Snack Stand, which offers sandwiches and light snacks.","Afternoon":"Take a leisurely hike on the Sentinel Dome and Taft Point trails for more incredible vistas.","Dinner":"Return to the campsite for a hearty campfire dinner. Consider grilling burgers or hot dogs."}}]}}
  """
    },
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Yellowstone National Park. They will be traveling in a group. They leave on 2024-10-06 and return home on 2024-10-08. They prefer to stay in campsites. They enjoy cycling, tours, rafting. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","trip_start":"2024-10-06","trip_end":"2024-10-08","num_days":"3","traveling_with":"in a group","lodging":"campsites","adventure":"cycling, tours, rafting","itinerary":[{{"Day 1":"Arrival and Camp Setup (2024-10-06)","Morning":"Arrive at Yellowstone National Park and check into your reserved campsite at Madison Campground.","Afternoon":"Set up your campsite and explore the nearby area. Take a short walk along the Madison River to get acquainted with the surroundings.","Dinner":"Dinner at the campsite. Prepare a meal using your camp stove or grill, enjoying the peaceful riverside setting."}},{{"Day 2":"Geysers and Hot Springs (2024-10-07)","Morning":"Visit the iconic Old Faithful Geyser and explore the Upper Geyser Basin, home to many of the park's geothermal features.","Lunch":"Dine at the Old Faithful Inn Dining Room, offering a variety of delicious meals in a historic setting.","Afternoon":"Take a guided tour of the Midway Geyser Basin, including the stunning Grand Prismatic Spring.","Dinner":"Return to the campsite for a campfire dinner. Consider making foil packet meals with vegetables and sausage."}},{{"Day 3":"Cycling in the Lamar Valley (2024-10-08)","Morning":"Head to Lamar Valley, known for its wildlife viewing opportunities. Enjoy a cycling tour through the valley, keeping an eye out for bison, elk, and wolves.","Lunch":"Picnic lunch in Lamar Valley. Pack sandwiches, snacks, and drinks.","Afternoon":"Continue your cycling adventure or take a short hike to Trout Lake.","Dinner":"Drive to Cooke City for dinner at the Beartooth Café, known for its hearty meals and rustic charm."}}]}}
  """
    },
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Haleakala National Park . They will be traveling with one partner. They leave on 2025-02-09 and return home on 2025-02-12. They prefer to stay in hotels, bed & breakfasts. They enjoy birding, hiking, swimming. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","trip_start":"2024-10-06","trip_end":"2024-10-08","num_days":"3","traveling_with":"in a group","lodging":"campsites","adventure":"cycling, tours, rafting","itinerary":[{{"Day 1":"Arrival and Sunset at Haleakalā (2025-02-09)","Morning":"Arrive in Maui and check into your hotel. Consider staying at Kula Lodge & Restaurant for its proximity to Haleakalā and cozy accommodations.","Afternoon":"Settle into your room and take a short rest.","Late Afternoon":"Drive up to Haleakalā National Park for a breathtaking sunset view from the summit. The panoramic vistas are stunning as the sun sets over the horizon.","Dinner":"Dine at Kula Lodge Restaurant, enjoying locally-sourced dishes and beautiful mountain views."}},{{"Day 2":"Sunrise, Birding, and Coastal Exploration (2025-02-10)","Early Morning":"Wake up early and drive to the Haleakalā summit to witness the sunrise. The experience is magical and well worth the early start.","Morning":"After sunrise, head to Hosmer Grove for birding. This area is home to many native Hawaiian birds, including the ‘Apapane and ‘I‘iwi.","Lunch":"Pack a picnic lunch to enjoy at Hosmer Grove or head back to Kula for a meal at La Provence, a charming bakery and café.","Afternoon":"Drive to the Kipahulu District of Haleakalā National Park. Hike the Pipiwai Trail to Waimoku Falls, a scenic 4-mile round-trip hike through lush forests and past bamboo groves.","Dinner":"Drive back to Hana and dine at Hana Ranch Restaurant, which offers fresh, farm-to-table Hawaiian cuisine."}},{{"Day 3":"Swimming and Exploring the Park (2025-02-11)","Morning":"Drive to the Seven Sacred Pools (ʻOheʻo Gulch) in the Kipahulu District. Enjoy a refreshing swim in the beautiful pools and explore the surrounding area.","Lunch":"Have lunch at the nearby Hāna Picnic Lunch Company, which offers delicious takeout options.","Afternoon":"Return to the main section of Haleakalā National Park and explore the Sliding Sands Trail (Keoneheʻeheʻe Trail). Hike as far as you are comfortable into the volcanic crater and experience the otherworldly landscape.","Dinner":"Drive to Paia and dine at Mama’s Fish House, renowned for its fresh seafood and unique Hawaiian ambiance."}}]}}
  """
    }
  ]
  
  example_prompt = PromptTemplate.from_template(
  template =
"""
{prompt}\n{response}
"""
    )
  #log.info(example_prompt.format(**examples[0]))
  
  few_shot_prompt = FewShotPromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
    suffix = "{input}",
    input_variables = ["input"],
  )
  
  return few_shot_prompt.format(input = "You are a trip planner who plans fun and memorable trips. The user is planning a trip to " + form_data["location"] + ". They will be traveling " + form_data["traveling_with_list"] + ". They leave on " + form_data["trip_start"] + " and return home on " + form_data["trip_end"] +". They prefer to stay in " + form_data["lodging_list"] + ". They enjoy " + form_data["adventure_list"] + ". Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal. You are also a backend data processor that is part of our app's programmatic workflow. Output the itenerary as only JSON with no text before or after the JSON.")
  
  
  
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
  # log.info(request.form)
  traveling_with_list = ", ".join(request.form.getlist("traveling-with"))
  lodging_list = ", ".join(request.form.getlist("lodging"))
  adventure_list = ", ".join(request.form.getlist("adventure"))
  
  cleaned_form_data = {
        "trip_name": request.form["trip-name"],
        "location": request.form["location-search"],
        "trip_start": request.form["trip-start"],
        "trip_end": request.form["trip-end"],
        "traveling_with_list": traveling_with_list,
        "lodging_list": lodging_list,
        "adventure_list": adventure_list,
        
    }
  # log.info(cleaned_form_data)
  # print(cleaned_form_data)
  
  prompt = build_new_trip_prompt(cleaned_form_data)
  
  response = llm.invoke(prompt)
  log.info(response)
  
  return render_template("view-trip.html")
    
# Run the flask server
if __name__ == "__main__":
    app.run()
