from flask import Flask, render_template, request
import logging
from datetime import datetime
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
import json

# Initialize the OpenAI language model
llm = OpenAI(
  max_tokens = -1 # This is not recomeended for producation.
)

# Initialize the output parser
parser = JsonOutputParser()

# app will run at: http://127.0.0.1:5000/

# Initialize logging
logging.basicConfig(filename="app.log", level=logging.INFO)
log = logging.getLogger("app")

# Initialize the Flask application
app = Flask(__name__)

# Define the function to build the new trip prompt
def build_new_trip_prompt_template():
  examples = [
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Yosemite National Park. They are traveling with one partner, in a group. They leave on 2024-08-18 and return home on 2024-08-20. They prefer to stay in campsites. They enjoy hiking, swimming. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yosemite Trip 2024","location":"Yosemite National Park","trip_start":"2024-08-18","trip_end":"2024-08-20","num_days":"3","traveling_with":"solo, with kids","lodging":"campsites","adventure":"hiking, swimming","itinerary":[{{"day": "1","date": "2024-08-18 - Arrival and Camp Setup ","morning":"Arrive at Yosemite National Park and check into your reserved campsite at Upper Pines Campground.","afternoon":"Set up your campsite and explore the area. Take a short hike to Mirror Lake to acclimate to the surroundings.","evening":"Dinner at the campsite with a view of Half Dome. Prepare a meal using your camp stove or grill."}},{{"day": "2", "date":"2024-08-19 - Exploring Yosemite Valley","morning":"Start the day with a hike on the Mist Trail to Vernal Fall and Nevada Fall. This is a moderately challenging hike with spectacular waterfall views.","Lunch":"Picnic lunch by the Merced River. Pack sandwiches, snacks, and drinks.","afternoon":"Visit the Yosemite Valley Visitor Center to learn about the park's history and geology.","evening":"Dine at the Yosemite Valley Lodge's Base Camp Eatery for a variety of casual dining options."}}]}}
  """
    },
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Yellowstone National Park. They will be traveling in a group. They leave on 2024-10-06 and return home on 2024-10-08. They prefer to stay in campsites. They enjoy cycling, tours, rafting. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","trip_start":"2024-10-06","trip_end":"2024-10-08","num_days":"3","traveling_with":"in a group","lodging":"campsites","adventure":"cycling, tours, rafting","itinerary":[{{"day": "1", "date":"2024-10-06 - Arrival and Camp Setup","morning":"Arrive at Yellowstone National Park and check into your reserved campsite at Madison Campground.","afternoon":"Set up your campsite and explore the nearby area. Take a short walk along the Madison River to get acquainted with the surroundings.","evening":"Dinner at the campsite. Prepare a meal using your camp stove or grill, enjoying the peaceful riverside setting."}},{{"day": "2", "date":"2024-10-07 - Geysers and Hot Springs ","morning":"Visit the iconic Old Faithful Geyser and explore the Upper Geyser Basin, home to many of the park's geothermal features.","Lunch":"Dine at the Old Faithful Inn Dining Room, offering a variety of delicious meals in a historic setting.","afternoon":"Take a guided tour of the Midway Geyser Basin, including the stunning Grand Prismatic Spring.","evening":"Return to the campsite for a campfire dinner. Consider making foil packet meals with vegetables and sausage."}}]}}
  """
    },
    {
      "prompt":
  """
  You are a trip planner who plans fun and memorable trips. The user is planning a trip to Haleakala National Park . They will be traveling with one partner. They leave on 2025-02-09 and return home on 2025-02-12. They prefer to stay in hotels, bed & breakfasts. They enjoy birding, hiking, swimming. Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal.
  """,
      "response":
  """
  {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","trip_start":"2024-10-06","trip_end":"2024-10-08","num_days":"3","traveling_with":"in a group","lodging":"campsites","adventure":"cycling, tours, rafting","itinerary":[{{"day": "1", "date":"2025-02-09 - Arrival and Sunset at Haleakalā","morning":"Arrive in Maui and check into your hotel. Consider staying at Kula Lodge & Restaurant for its proximity to Haleakalā and cozy accommodations.","afternoon":"Settle into your room and take a short rest.","Late Afternoon":"Drive up to Haleakalā National Park for a breathtaking sunset view from the summit. The panoramic vistas are stunning as the sun sets over the horizon.","evening":"Dine at Kula Lodge Restaurant, enjoying locally-sourced dishes and beautiful mountain views."}},{{"day": "2", "date":"2025-02-10 - Sunrise, Birding, and Coastal Exploration","Early Morning":"Wake up early and drive to the Haleakalā summit to witness the sunrise. The experience is magical and well worth the early start.","morning":"After sunrise, head to Hosmer Grove for birding. This area is home to many native Hawaiian birds, including the ‘Apapane and ‘I‘iwi.","Lunch":"Pack a picnic lunch to enjoy at Hosmer Grove or head back to Kula for a meal at La Provence, a charming bakery and café.","afternoon":"Drive to the Kipahulu District of Haleakalā National Park. Hike the Pipiwai Trail to Waimoku Falls, a scenic 4-mile round-trip hike through lush forests and past bamboo groves.","evening":"Drive back to Hana and dine at Hana Ranch Restaurant, which offers fresh, farm-to-table Hawaiian cuisine."}}]}}
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
    suffix = "This trip is to {location} between {trip_start} and {trip_end}. This person will be traveling {traveling_with} and would like to stay in {lodging}. They want to {adventure}. Create a daily itinerary for this trip using this information. You are a backend data processor that is part of our site's programmatic workflow. Output the itinerary as only JSON with no text before or after the JSON.",
    input_variables = ["location", "trip_start", "trip_end", "traveling_with", "lodging", "adventure"],
  )
  return few_shot_prompt
  # return few_shot_prompt.format(input = "You are a trip planner who plans fun and memorable trips. The user is planning a trip to " + form_data["location"] + ". They will be traveling " + form_data["traveling_with_list"] + ". They leave on " + form_data["trip_start"] + " and return home on " + form_data["trip_end"] +". They prefer to stay in " + form_data["lodging_list"] + ". They enjoy " + form_data["adventure_list"] + ". Build out an itenerary broken out by day. Each day should include at least one activity and one restaurant meal. You are also a backend data processor that is part of our app's programmatic workflow. Output the itenerary as only JSON with no text before or after the JSON.")
  
# Prompt for the typical weather conditions  
def build_weather_prompt_template():
  examples = [
    {
      "prompt": 
        """
        Update the following JSON object to include typical weather conditions for the trip based on the values of trip_start, trip_end, and location. Keep the object exactly as it is, and add a key / value pair to the JSON, with the key being typical_weather and the value being a string describing the typical weather for the time period. Add this key / value pair after the key / value pair with a key of location. You are a backend data processor that is part of our site's programmatic workflow. Output the updated itinerary as only JSON with no text before or after the JSON. Do not include the words updated itinerary at the beginning of your response. {{"trip_name":"Acadia Trip 2024","location":"Acadia National Park","trip_start":"2024-09-13","trip_end":"2024-09-15","num_days":"3","traveling_with":"partner","lodging":"hotels","adventure":"guided tours, swimming","itinerary":[{{"day":"1","date":"2024-09-13 - Arrival and Exploration of the Island","morning":"Arrive at Acadia National Park and check into your hotel.","afternoon":"Take a guided tour of the island, learning about its history, geology, and wildlife.","evening":"Dine at The Burning Tree Restaurant, known for its fresh seafood and beautiful views of the harbor."}},{{"day":"2","date":"2024-09-14 - Hiking and Sightseeing","morning":"Start the day with a hike on the Precipice Trail, a challenging but rewarding trail with stunning views of the coast.","Lunch":"Picnic lunch at Sand Beach, a beautiful stretch of sand surrounded by rocky cliffs.","afternoon":"Explore the Jordan Pond area and take a scenic drive along the Park Loop Road.","evening":"Dine at The Thirsty Whale Tavern, a local favorite for its delicious pub-style food and lively atmosphere."}}]}}
        """,
      "response":
        """
        {{"trip_name":"Acadia Trip 2024","location":"Acadia National Park","typical_weather":"In September, the weather at Acadia National Park in Maine is typically in the 60s for highs and the high 40s for lows. The fall season brings cooler weather, and the crowds from summer have thinned out.","trip_start":"2024-09-13","trip_end":"2024-09-15","num_days":"3","traveling_with":"partner","lodging":"hotels","adventure":"guided tours, rafting","itinerary":[{{"day":"1","date":"2024-09-13 - Arrival and Exploration of the Island","morning":"Arrive at Acadia National Park and check into your hotel.","afternoon":"Take a guided tour of the island, learning about its history, geology, and wildlife.","evening":"Dine at The Burning Tree Restaurant, known for its fresh seafood and beautiful views of the harbor."}},{{"day":"2","date":"2024-09-14 - Hiking and Sightseeing","morning":"Start the day with a hike on the Precipice Trail, a challenging but rewarding trail with stunning views of the coast.","Lunch":"Picnic lunch at Sand Beach, a beautiful stretch of sand surrounded by rocky cliffs.","afternoon":"Explore the Jordan Pond area and take a scenic drive along the Park Loop Road.","evening":"Dine at The Thirsty Whale Tavern, a local favorite for its delicious pub-style food and lively atmosphere."}}]}}
        """,
    },
    {
      "prompt": 
        """
        Update the following JSON object to include typical weather conditions for the trip based on the values of trip_start, trip_end, and location. Keep the object exactly as it is, and add a key / value pair to the JSON, with the key being typical_weather and the value being a string describing the typical weather for the time period. Add this key / value pair after the key / value pair with a key of location. You are a backend data processor that is part of our site's programmatic workflow. Output the updated itinerary as only JSON with no text before or after the JSON. Do not include the words updated itinerary at the beginning of your response. {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","trip_start":"2024-09-11","trip_end":"2024-09-13","num_days":"3","traveling_with":"partner","lodging":"hotels","activity":"swimming","itinerary":[{{"day":"1","date":"2024-09-11 - Arrival and Exploration","morning":"Arrive at Yellowstone National Park and check into your hotel. Recommended hotels include Old Faithful Inn, Lake Yellowstone Hotel, or Canyon Lodge.","afternoon":"Settle into your room and explore the nearby area. Take a walk around the Old Faithful Geyser or visit the Grand Canyon of the Yellowstone.","evening":"Dinner at the Old Faithful Inn Dining Room, offering a variety of delicious meals in a historic setting."}},{{"day":"2","date":"2024-09-12 - Swimming in Yellowstone Lake","morning":"Start the day with a refreshing swim in Yellowstone Lake, the largest high elevation lake in North America.","Lunch":"Picnic lunch by the lake. Pack sandwiches, snacks, and drinks.","afternoon":"Take a scenic drive around the lake and make stops at various viewpoints and hiking trails."}}]}} 
        """,
    
      "response":
      """
      {{"trip_name":"Yellowstone Trip 2024","location":"Yellowstone National Park","typical_weather":"The weather in Yellowstone National Park in September is usually warm with variable temperatures, but it can still be chilly at night","trip_start":"2024-09-11","trip_end":"2024-09-13","num_days":"3","traveling_with":"partner","lodging":"hotels","activity":"swimming","itinerary":[{{"day":"1","date":"2024-09-11 - Arrival and Exploration","morning":"Arrive at Yellowstone National Park and check into your hotel. Recommended hotels include Old Faithful Inn, Lake Yellowstone Hotel, or Canyon Lodge.","afternoon":"Settle into your room and explore the nearby area. Take a walk around the Old Faithful Geyser or visit the Grand Canyon of the Yellowstone.","evening":"Dinner at the Old Faithful Inn Dining Room, offering a variety of delicious meals in a historic setting."}},{{"day":"2","date":"2024-09-12 - Swimming in Yellowstone Lake","morning":"Start the day with a refreshing swim in Yellowstone Lake, the largest high elevation lake in North America.","Lunch":"Picnic lunch by the lake. Pack sandwiches, snacks, and drinks.","afternoon":"Take a scenic drive around the lake and make stops at various viewpoints and hiking trails."}}]}}
      """
    }
  ]   
  
  example_prompt = PromptTemplate.from_template(
    template =
"""
{prompt}\n{response}
"""
    )
  few_shot_prompt = FewShotPromptTemplate(
        examples = examples,
        example_prompt = example_prompt,
        suffix = "Update the following JSON object to include typical weather conditions for the trip based on the values of trip_start, trip_end, and location. Keep the object exactly as it is, and add a key / value pair to the JSON, with the key being typical_weather and the value being a string describing the typical weather for the time period. Add this key / value pair after the key / value pair with a key of location. You are a backend data processor that is part of our site's programmatic workflow. Output the itinerary as only JSON with no text before or after the JSON. {input}",
        input_variables = ["input"],
    )
  return few_shot_prompt

  
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
  
  prompt = build_new_trip_prompt_template()
  
  chain = prompt | llm | parser
  
  output = chain.invoke({
        "location": request.form["location-search"],
        "trip_start": request.form["trip-start"],
        "trip_end": request.form["trip-end"],
        "traveling_with": traveling_with_list,
        "lodging": lodging_list,
        "adventure": adventure_list,
        "trip_name": request.form["trip-name"],
        
    })
  
  log.info(output)
  
  prompt2 = build_weather_prompt_template()
  
  chain2 = prompt2 | llm | parser
  
  output_str = json.dumps(output)
  
  output2 = chain2.invoke({"input": output_str})
  
  log.info(output2)
  
  return render_template("view-trip.html", output = output2)
 
  
  # log.info(cleaned_form_data)
  # print(cleaned_form_data)
     
  
  #log.info(response)
  # delete comments below after testing changes:
  #response = llm.invoke(prompt)
  #output = parser.parse(response)
  
  
  
  #return render_template("view-trip.html", output = output)
    
# Run the flask server
if __name__ == "__main__":
    app.run()
