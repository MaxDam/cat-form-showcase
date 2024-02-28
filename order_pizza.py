from typing import List, Dict
from pydantic import BaseModel, Field
from cat.log import log
from cat.experimental.form import form, CatForm
import random

class PizzaOrder(BaseModel):
    pizza_type: str
    name:       str
    address:    str
    phone:      str


@form
class PizzaForm(CatForm):

    description = "Pizza Order"
    model_class = PizzaOrder
    ask_confirm = False
    start_examples = [
        "order a pizza",
        "I want pizza",
        "I would like order a pizza"
    ]
    stop_examples = [
        "I would like to exit the module",
        "I no longer want to continue filling out the form",
        "You go out",
        "Return to normal conversation",
        "Stop and go out"
    ]

    form_chat_history = []


    # Check exit intent with examples
    def check_exit_intent(self) -> bool:

        # Stop examples
        stop_example_prompt = ""
        if len(self.stop_examples) > 0:
            stop_example_prompt = "Examples of User Message:"
            stop_example_prompt += "".join(f"\n- {stop_example}" for stop_example in self.stop_examples)

        # Get user message
        user_message = self.cat.working_memory["user_message_json"]["text"]

        # Check exit prompt
        check_exit_prompt = \
f"""Your task is to produce a JSON representing whether a user wants to exit or not.
JSON must be in this format:
```json
{{
    "exit": // type boolean, must be `true` or `false`
}}
```

{stop_example_prompt}

###User Message:
{user_message}

###AI Response:
JSON:
```json
{{
    "exit":"""

        # Print confirm prompt
        print(check_exit_prompt)

        # Queries the LLM and check if user is agree or not
        response = self.cat.llm(check_exit_prompt, stream=True)
        return "true" in response.lower()


    # Reprocessing of the response message to the user and save local chat history
    def message(self):
        response_message = super().message()
        print(f"message: {response_message}")

        # Add dialogue interaction to chat history
        user_message = self.cat.working_memory["user_message_json"]["text"]
        self.form_chat_history.append({"who": "Human", "message": user_message,     "why": {}})
        self.form_chat_history.append({"who": "AI",    "message": response_message, "why": {}})

        prompt = \
f"""You are a helpful assistant and your task is to request missing information
from the user or present any errors, but one piece of information at a time.
For example, if the user's name and surname are missing, but you have their address,
you should respond: "I have already obtained your address, could you now provide me with your name?"
Given the following information:

{response_message}

Please request the missing information from the user in the specified format.
For instance, if the missing details are address and phone, you should say:
"I see that you have provided a name. However,
I still need your surname and email. Could you start by giving me your name?" 
"""
        
        response = self.cat.llm(prompt, stream=True)
        return response


    # Local chat history
    def stringify_convo_history(self):

        user_message = self.cat.working_memory["user_message_json"]["text"]
        form_chat_history = self.form_chat_history[-10:] # last n messages

        # stringify history
        history = ""
        for turn in form_chat_history:
            history += f"\n - {turn['who']}: {turn['message']}"
        history += f"Human: {user_message}"

        return history


    # Submit form
    def submit(self, form_data):  
        result = "<h3>PIZZA CHALLENGE - ORDER COMPLETED<h3><br>" 
        result += "<table border=0>"
        result += "<tr>"
        result += "   <td>Pizza Type</td>"
        result += f"  <td>{form_data['pizza_type']}</td>"
        result += "</tr>"
        result += "<tr>"
        result += "   <td>Name</td>"
        result += f"  <td>{form_data['name']}</td>"
        result += "</tr>"
        result += "<tr>"
        result += "   <td>Address</td>"
        result += f"  <td>{form_data['address']}</td>"
        result += "</tr>"
        result += "<tr>"
        result += "   <td>Phone Number</td>"
        result += f"  <td>{form_data['phone']}</td>"
        result += "</tr>"
        result += "</table>"
        result += "<br>"                                                                                                     
        result += "Thanks for your order.. your pizza is on its way!"
        result += "<br><br>"
        result += f"<img style='width:400px' src='https://maxdam.github.io/cat-pizza-challenge/img/order/pizza{random.randint(0, 6)}.jpg'>"
        return result
