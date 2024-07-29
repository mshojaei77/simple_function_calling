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
                tool_choice="auto"
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
