import gradio as gr
import os
import io
import requests, json
from PIL import Image
import base64
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Load the translation model (Turkish to English)
API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-tr-en"

headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }

# Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=os.environ['HF_API_TTI_STABILITY_AI']):
    data = {"inputs": inputs}
    if parameters is not None:
        data.update({"parameters": parameters})

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    
    # Check the content type of the response
    content_type = response.headers.get('Content-Type', '')
    print(content_type)
    if 'application/json' in content_type:
        return json.loads(response.content.decode("utf-8"))
    elif 'image/' in content_type:
        return response.content  # return raw image data

    response.raise_for_status()  # raise an error for unexpected content types

# A helper function to convert the PIL image to base64 
def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

# Translation function
def translate_to_english(text):
    try:
        # Translate the input from Turkish to English
        translation = query({"inputs": text})
        print(translation)
        translated_text = translation[0]['translation_text']
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # If translation fails, return original text

# Main generation function with translation
def generate(prompt, negative_prompt, steps, guidance, width, height):
    # Translate the prompt to English if it's in Turkish
    translated_prompt = translate_to_english(prompt)
    print(f"Translated Prompt: {translated_prompt}")
    
    params = {
        "negative_prompt": negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "width": width,
        "height": height
    }
    
    output = get_completion(translated_prompt, params)
    
    # Check if the output is an image (bytes) or JSON (dict)
    if isinstance(output, dict):
        raise ValueError("Expected an image but received JSON: {}".format(output))
    
    # If output is raw image data, convert it to a PIL image
    result_image = Image.open(io.BytesIO(output))
    return (translated_prompt, result_image)

with gr.Blocks() as demo:
    gr.Markdown("## Image Generation with Turkish Inputs")
    gr.Markdown("### [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [`Helsinki-NLP/opus-mt-tr-en`](https://huggingface.co/Helsinki-NLP/opus-mt-tr-en) models work under the hood!")
    
    with gr.Row():
        with gr.Column(scale=4):
            prompt = gr.Textbox(label="Your prompt (in Turkish or English)") # Accept Turkish or English input
        with gr.Column(scale=1, min_width=50):
            btn = gr.Button("Submit")
    
    with gr.Accordion("Advanced options", open=False):
        negative_prompt = gr.Textbox(label="Negative prompt")
        with gr.Row():
            with gr.Column():
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=100, value=50,
                                  info="In how many steps will the denoiser denoise the image?")
                guidance = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, value=12,
                                     info="Controls how much the text prompt influences the result")
            with gr.Column():
                width = gr.Slider(label="Width", minimum=64, maximum=1024, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=64, maximum=1024, step=64, value=1024)
    translated_text = gr.Textbox(label="Translated text")
    output = gr.Image(label="Result")
            
    btn.click(fn=generate, inputs=[prompt, negative_prompt, steps, guidance, width, height], outputs=[translated_text, output])

gr.close_all()
demo.launch(share=True)