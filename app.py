from flask import Flask, render_template, request
import logging
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import create_json_chat_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import StructuredTool
from langchain import hub

# Initialize the OpenAI language model
llm = ChatOpenAI(
  max_tokens = 4000, model="gpt-3.5-turbo", temperature=0.5
)

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
    """Handles the form submission to view the generated trip itinerary."""
    # Extract form data
    location = request.form["location-search"]
    trip_start = request.form["trip-start"]
    trip_end = request.form["trip-end"]
    traveling_with = ", ".join(request.form.getlist("traveling-with"))
    lodging = ", ".join(request.form.getlist("lodging"))
    adventure = ", ".join(request.form.getlist("adventure"))
    
    # Create the input string with the user's unique trip information
    input_data = generate_trip_input(location, trip_start, trip_end, traveling_with, lodging, adventure)

    print('input_data: \n', input_data, '\n')
    
    # Create a tool for the agent to use that utilizes Wikipedia's run function
    wikipedia_tool = create_wikipedia_tool()
    
    # Pull a tool prompt template from the hub
    prompt = hub.pull("hwchase17/react-chat-json")
    
    # Create our agent that will utilize tools and return JSON
    agent = create_json_chat_agent(llm=llm, tools=[wikipedia_tool], prompt=prompt)
    
    # Create a runnable instance of the agent
    agent_executor = AgentExecutor(agent=agent, tools=[wikipedia_tool], verbose=True, handle_parsing_errors="The output from the LLM could not be parsed or is incomplete."
    )
    
    # Invoke the agent with the input data
    response = agent_executor.invoke({"input": input_data})
    
    log.info(response["output"])

    # Render the response on the view-trip.html page
    return render_template("view-trip.html", output=response["output"])
    

# Define generate trip function
def generate_trip_input(location, trip_start, trip_end, traveling_with, lodging, adventure):
    """
    Generates a structured input string for the trip planning agent.
    """
    return f"""
    Create an itinerary for a trip to {location}.
    The trip starts on: {trip_start}
    The trip ends on: {trip_end}
    I will be traveling with {traveling_with}
    I would like to stay in {lodging}
    I would like to do the following activities: {adventure}
 
    Please generate a complete and detailed trip itinerary with the following JSON data structure:
 
    {{
        "trip_name": "String - Name of the trip",
        "location": "String - Location of the trip",
        "trip_start": "String - Start date of the trip",
        "trip_end": "String - End date of the trip",
        "typical_weather": "String - Description of typical weather for the trip",
        "traveling_with": "String - Description of travel companions",
        "lodging": "String - Description of lodging arrangements",
        "adventure": "String - Description of planned activities",
        "itinerary": [
            {{
                "day": "Integer - Day number",
                "date": "String - Date of this day",
                "morning": "String - Description of morning activities",
                "afternoon": "String - Description of afternoon activities",
                "evening": "String - Description of evening activities"
            }}
        ],
        "important_things_to_know": "String - Any important things to know about the park being visited."
    }}
 
    The trip should be appropriate for those listed as traveling, themed around the interests specified, and that last for the entire specified duration of the trip.
    Include realistic and varied activities for each day, considering the location, hours of operation, and typical weather.
    Make sure all fields are filled with appropriate and engaging content.
    Include descriptive information about each day's activities and destination.
    Respond only with a valid parseable JSON object representing the itinerary.
    """

# Create Wiki tool function
def create_wikipedia_tool():
    """
    Creates a built-in langchain tool for querying Wikipedia.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return StructuredTool.from_function(
        func=wikipedia.run,
        name="Wikipedia",
        description="Useful for Wikipedia searches about national parks."
    )  
  
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
