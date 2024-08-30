## It's a course playground for "Building Generative AI Applications with Gradio" course from dlai.

### Check course website:
https://learn.deeplearning.ai/courses/huggingface-gradio

## Models used:
### L1: NLP TASK INTERFACE
* [`dslim/bert-base-NER`](https://huggingface.co/dslim/bert-base-NER)
* [`shleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6)

NER(named-entity-recognition) model `dslim/bert-base-NER` is hosted at Huggingface spaces at following URL: https://huggingface.co/spaces/dolphinium/bert_ner 

### L2: IMAGE CAPTIONING APP
* [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
### L3: IMAGE GENERATION APP
* [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)

Image generation model [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) is hosted at Huggingface spaces at following URL: https://huggingface.co/spaces/dolphinium/stable_diffusion_image_gen
### L4: DESCRIBE AND GENERATE GAME
* [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base)
* [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main)


Describe and Regenerate model [`TTI STABLE DIFFUSION`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) [`ITT BLIP-large`](https://huggingface.co/Salesforce/blip-image-captioning-large) is hosted at Huggingface spaces at following URL: 
https://huggingface.co/spaces/dolphinium/Describe_and_Regenerate. 
* It is a demonstration of multi model usage on HuggingFace Spaces and Gradio

### L5: CHAT WITH ANY LLM
* [`falcon-40b-instruct`](https://huggingface.co/tiiuae/falcon-40b-instruct) (used on course notebook)

since serverless inference api support has been disabled for falcon, I tried these two models:

* [`mistralai/Mistral-Nemo-Instruct-2407`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)
* [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)(requires hugging face pro subscription for serverless inference api)