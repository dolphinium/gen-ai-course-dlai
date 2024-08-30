import os
import io
from PIL import Image
import base64 
import gradio as gr
import requests, json
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

#text-to-image
TTI_ENDPOINT = os.environ['HF_API_TTI_STABILITY_AI']
#image-to-text
ITT_ENDPOINT = os.environ['HF_API_ITT_BASE']

# Text-to-image endpoint
def get_completion(inputs, parameters=None, ENDPOINT_URL=TTI_ENDPOINT):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
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


#Bringing the functions from lessons 3 and 4!
def image_to_base64_str(pil_image):
    byte_arr = io.BytesIO()
    pil_image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()
    return str(base64.b64encode(byte_arr).decode('utf-8'))

def base64_to_pil(img_base64):
    base64_decoded = base64.b64decode(img_base64)
    byte_stream = io.BytesIO(base64_decoded)
    pil_image = Image.open(byte_stream)
    return pil_image

def captioner(image):
    base64_image = image_to_base64_str(image)
    result = get_completion(base64_image, None, ITT_ENDPOINT)
    return result[0]['generated_text']

def generate(prompt):
    output = get_completion(prompt)
    
    # Check if the output is an image (bytes) or JSON (dict)
    if isinstance(output, dict):
        raise ValueError("Expected an image but received JSON: {}".format(output))
    
    # If output is raw image data, convert it to a PIL image
    result_image = Image.open(io.BytesIO(output))
    return result_image


def caption_and_generate(image):
    caption = captioner(image)
    image = generate(caption)
    return [caption, image]

with gr.Blocks() as demo:
    gr.Markdown("# Describe-and-Generate game 🖍️")
    image_upload = gr.Image(label="Your first image",type="pil")
    btn_all = gr.Button("Caption and generate")
    caption = gr.Textbox(label="Generated caption")
    image_output = gr.Image(label="Generated Image")

    btn_all.click(fn=caption_and_generate, inputs=[image_upload], outputs=[caption, image_output])

gr.close_all()
demo.launch(share=True)