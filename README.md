# Connecting Large Language Models to External Tools Using Function Calling: A Beginner's Guide

Welcome to this beginner-friendly tutorial on connecting large language models (LLMs) to external tools using function calling. In this guide, you'll learn how to use OpenAI's Chat Completions API to integrate functions into a chatbot that can interact with external APIs or perform specific tasks.

## Introduction

Function calling allows LLMs to interact with external tools or APIs by generating structured JSON that describes function calls. This capability helps transform natural language queries into actionable API calls. For example, a model could determine that a user wants to know the current weather and generate JSON to call a weather API.

### Key Points:

- **Function Calling:** LLMs generate JSON to describe function calls, which your code executes.
- **Supported Models:** gpt-4o, gpt-4-turbo, and others.
- **Common Use Cases:** Answering questions, converting natural language into API calls, and extracting structured data.
- **Risks:** Always validate and confirm function calls to avoid unintended actions.

## Step-by-Step Tutorial

### Step 1: Set Up Your Development Environment

1. **Install Required Libraries**

   You need the following libraries:

   - `openai`: To communicate with the OpenAI API.
   - `python-dotenv`: To manage environment variables securely.

   Install them using pip:

   ```bash
   pip install openai python-dotenv
   ```

2. **Create a `.env` File**

   This file will store your OpenAI API key. Create a `.env` file in the root of your project with the following content:

   ```plaintext
   OPENAI_API_KEY=your_openai_api_key
   ```

   Replace `your_openai_api_key` with your actual OpenAI API key.

### Step 2: Create a Custom Function

1. **Define a Function for Weather Information**

   Create a file named `functions.py` and define a function to retrieve weather information. This is a dummy function that simulates weather data:

   ```python
   import json

   def get_current_weather(location, unit="fahrenheit"):
       """Get the current weather in a given location"""
       if "tokyo" in location.lower():
           return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
       elif "san francisco" in location.lower():
           return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
       elif "paris" in location.lower():
           return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
       else:
           return json.dumps({"location": location, "temperature": "unknown"})
   ```

### Step 3: Define Tool Specifications

1. **Create a `tools.json` File**

   This file describes the tools (functions) your chatbot can use. Create a file named `tools.json` with the following content:

   ```json
   [
       {
           "type": "function",
           "function": {
               "name": "get_current_weather",
               "description": "Get the current weather in a given location",
               "parameters": {
                   "type": "object",
                   "properties": {
                       "location": {
                           "type": "string",
                           "description": "The city and state, e.g. San Francisco, CA"
                       },
                       "unit": {
                           "type": "string",
                           "enum": ["celsius", "fahrenheit"]
                       }
                   },
                   "required": ["location"]
               }
           }
       }
   ]
   ```

   This JSON specifies that the `get_current_weather` function requires a `location` parameter and optionally a `unit` parameter.

### Step 4: Implement the Chatbot

1. **Create the Main Script**

   Create a file named `chatbot.py` and add the following code to implement the chatbot:

   ```python
   import os
   import json
   from openai import OpenAI
   from dotenv import load_dotenv
   from functions import get_current_weather

   class Chatbot:
       def __init__(self):
           load_dotenv()
           self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
           self.messages = []
           self.tools = self._load_json('tools.json')
           self.functions = {"get_current_weather": get_current_weather}

       def _load_json(self, file_path):
           try:
               with open(file_path, 'r', encoding='utf-8') as file:
                   return json.load(file)
           except Exception as e:
               print(f"Error loading {file_path}: {e}")
               return {}

       def send_message(self, content):
           self.messages.append({"role": "user", "content": content})

       def get_response(self, model="gpt-4o-mini"):
           try:
               response = self.client.chat.completions.create(
                   model=model,
                   messages=self.messages,
                   tools=self.tools,
                   tool_choice="auto"  # Model decides which functions to call
               )
               return response
           except Exception as e:
               print(f"Error getting response: {e}")
               return None

       def handle_tool_calls(self, tool_calls):
           for tool_call in tool_calls:
               func = self.functions.get(tool_call.function.name)
               if func:
                   args = json.loads(tool_call.function.arguments)
                   response = func(**args)
                   self.messages.append({
                       "tool_call_id": tool_call.id,
                       "role": "tool",
                       "name": tool_call.function.name,
                       "content": response
                   })
               else:
                   print(f"Function {tool_call.function.name} not available")

       def run_conversation(self, initial_message):
           self.send_message(initial_message)
           response = self.get_response()
           if response:
               response_message = response.choices[0].message
               tool_calls = response_message.tool_calls
               if tool_calls:
                   self.messages.append(response_message)
                   self.handle_tool_calls(tool_calls)
                   response = self.get_response()
               return response
           return None

   if __name__ == '__main__':
       bot = Chatbot()
       initial_message = "What's the weather like in San Francisco, Tokyo, and Paris?"
       response = bot.run_conversation(initial_message)
       content = response.choices[0].message.content
       if content:
           print(content)
       elif response:
           print(response)
       else:
           print("Failed to get a response.")
   ```

### Step 5: Run and Test the Chatbot

1. **Execute the Script**

   Run the chatbot script to see it in action:

   ```bash
   python chatbot.py
   ```

   The chatbot should output the current weather for San Francisco, Tokyo, and Paris based on the hardcoded responses in the `get_current_weather` function.

## Summary

In this tutorial, you have:

- Set up your environment and installed necessary libraries.
- Created a custom function to simulate retrieving weather information.
- Defined tool specifications using a JSON file.
- Implemented a chatbot that integrates with the OpenAI API and uses function calling.
- Tested the chatbot to ensure it retrieves and displays weather information.

### Additional Details:

- **Function Calling Behavior:** By default, the model decides when to call functions (`tool_choice: "auto"`). You can customize this behavior if needed.
- **Parallel Function Calling:** The model can make multiple function calls simultaneously, reducing round trips with the API.
- **Token Usage:** Functions count against the model's context limit. Keep this in mind if you have many functions or large documentation.

Feel free to expand this example with additional functions and tools, or integrate other APIs and services to enhance your chatbot's capabilities. Happy coding!
