#Draft of an llm agent system to assess image and text data about operating system state, send it to llama 3.2 90b vision, decide what actions to take
#writes code, tests and runs code gathering images and state data, assesses resulting state and success, then submits 
#it back to first agent in loop
#need to refine draft to  successfully ingest images into system prompt for multimodal analysis by groq llama
#need to refine the invocations to tools and execution between the second and third agents
import os
import json
import time
from dotenv import load_dotenv
from swarms import Agent
from groq import Groq
import pyautogui
from PIL import ImageGrab

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize Groq client
client = Groq(api_key=api_key)

# Define the Groq-based model
class GroqModel:
    def __init__(self, client):
        self.client = client

    def __call__(self, prompt):
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}],
                model="llama-3.2-90b-vision",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"

# Initialize the model
model = GroqModel(client=client)

# Serialize images with unique timestamps
def save_image(image):
    timestamp = int(time.time())
    filename = f"system_state_{timestamp}.png"
    image.save(filename)
    return filename

# Define agents
vision_understanding_agent = Agent(
    agent_name="Vision Understanding Agent",
    system_prompt="""
    You are a vision understanding agent specializing in analyzing system states. Your inputs include:
    - An image representing the current system state (screenshot of the GUI or terminal).
    - Text describing the current state or user instructions.

    Tasks:
    1. Analyze the image to extract key system information (e.g., open windows, errors, visible outputs).
    2. Combine image analysis with the text input to recommend the next steps.
    3. Output structured actions in JSON format:
        - Applications to open.
        - Commands to execute.
        - GUI interactions (e.g., clicks, typing).
    """,
    llm=model,
)

software_engineering_agent = Agent(
    agent_name="Software Engineering Agent",
    system_prompt="""
    You are an automation expert in interacting with GUI/IDE/terminal environments. Given a set of structured actions, your task is to:
    1. Execute the specified actions on the operating system:
        - Open applications.
        - Write and execute code in an IDE.
        - Perform web searches and scrape results.
    2. Capture the resulting system state:
        - Save a screenshot of the UI.
        - Extract any visible textual output.
    3. Return the new state as:
        - A serialized image path (screenshot).
        - Text data extracted from the UI.
    """,
    llm=model,
)

assessment_agent = Agent(
    agent_name="Assessment Agent",
    system_prompt="""
    You are responsible for assessing system states. Given the following inputs:
    - An image showing the resulting system state.
    - Text describing the outputs visible on the system.

    Tasks:
    1. Evaluate whether the previous actions achieved their goals.
    2. Provide structured feedback on success or failure.
    3. Suggest modifications or additional actions to achieve the task.
    """,
    llm=model,
)

# Function to capture the system state
def capture_system_state():
    screenshot = ImageGrab.grab()
    image_path = save_image(screenshot)
    # Example placeholder for UI text scraping
    scraped_text = "Placeholder text from UI"
    return image_path, scraped_text

# Infinite loop for agent orchestration
def process_task(initial_prompt):
    current_prompt = initial_prompt

    while True:
        # Vision Understanding Agent
        image_path, ui_text = capture_system_state()
        structured_actions = vision_understanding_agent(
            f"Analyze the following image and text:\nImage: {image_path}\nText: {ui_text}"
        )

        # Software Engineering Agent
        new_state = software_engineering_agent(structured_actions)

        # Parse new state (image + text)
        new_image_path = new_state.get("image_path")
        new_text_data = new_state.get("text_data")

        # Assessment Agent
        feedback = assessment_agent(
            f"Evaluate the system state:\nImage: {new_image_path}\nText: {new_text_data}"
        )

        # Update the prompt with feedback for the next iteration
        current_prompt = feedback
        print("Feedback for next iteration:", feedback)

# Example initial task
initial_prompt = "Build a Python program that lists all files in a directory, filters for `.txt` files, and saves the list to a new file."
process_task(initial_prompt)
