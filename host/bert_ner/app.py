import gradio as gr
import os
import requests, json
from dotenv import load_dotenv, find_dotenv

# Helper function

_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

#Summarization endpoint
def get_completion(inputs, parameters=None,ENDPOINT_URL=os.environ['HF_API_SUMMARY_BASE']): 
    headers = {
      "Authorization": f"Bearer {hf_api_key}",
      "Content-Type": "application/json"
    }
    data = { "inputs": inputs }
    if parameters is not None:
        data.update({"parameters": parameters})
    response = requests.request("POST",
                                ENDPOINT_URL, headers=headers,
                                data=json.dumps(data)
                               )
    return json.loads(response.content.decode("utf-8"))


def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens and token['entity_group'].startswith('I-') and merged_tokens[-1]['entity_group'].endswith(token['entity_group'][2:]):
            # If current token continues the  of the last one, merge them
            last_token = merged_tokens[-1]
            last_token['word'] += token['word'].replace('##', '')
            last_token['end'] = token['end']
            last_token['score'] = (last_token['score'] + token['score']) / 2
        else:
            # Otherwise, add the token to the list
            merged_tokens.append(token)

    return merged_tokens

HF_NER_API_URL = os.environ["HF_API_NER_BASE"]

def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=HF_NER_API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples=["The Roman Empire ruled the Mediterranean and much of Europe, Western Asia and North Africa. The Romans conquered most of this during the Republic, and it was ruled by emperors following Octavian's assumption of effective sole rule in 27 BC. The western empire collapsed in 476 AD, but the eastern empire lasted until the fall of Constantinople in 1453.",
                              "Hi, I'm Yunus. I recently graduated from Eskisehir Technical University with a degree in Computer Engineering. That's all for now. Happy coding ✌🏻"])

demo.launch(share=True)