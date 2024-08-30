import gradio as gr
import os
import requests 
from huggingface_hub import InferenceClient
import gradio as gr
from dotenv import load_dotenv, find_dotenv

requests.adapters.DEFAULT_TIMEOUT = 60
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

client = InferenceClient(
    "mistralai/Mistral-Nemo-Instruct-2407",
    token=hf_api_key,
)

def format_chat_prompt(message, chat_history, system_message):
    prompt = system_message
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def generate(formatted_prompt, slider, temperature):
    messages = [{"role": "user", "content": formatted_prompt}]
    output = ""
    for message in client.chat_completion(messages=messages, max_tokens=slider, temperature=temperature, stream=True):
        output += message.choices[0].delta.content
        yield output  # Stream the output to update the chatbot immediately

def respond(message, chat_history, system_message, temperature):
    formatted_prompt = format_chat_prompt(message, chat_history, system_message)
    bot_response_stream = generate(formatted_prompt, 1024, temperature)

    # Append the user's message to chat history before generating the bot's response
    chat_history.append((message, ""))

    for bot_message in bot_response_stream:
        chat_history[-1] = (message, bot_message)
        yield "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240)  # Just to fit the notebook
    msg = gr.Textbox(label="Prompt")
    with gr.Accordion("Advanced Options", open=False):
        system_msg = gr.Textbox(label="System Message", value="You are a helpful assistant.", placeholder="Set a custom system message.")
        temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.7, step=0.1)
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear chat")

    
    btn.click(respond, inputs=[msg, chatbot, system_msg, temperature], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system_msg, temperature], outputs=[msg, chatbot])  # Press enter to submit

gr.close_all()
demo.launch(share=True)