import os
import base64

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
import json
import time

api_key = os.getenv('GOOGLE_GEMINI_KEY')
model_name = 'gemini-2.5-pro'
client = genai.Client(api_key=api_key)


def pil_to_bytes(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return buffered.getvalue()


def request_vlm_sequence(prompt_template, prompt_content, images):
    '''
    prompt_content: dict
    images: List[PIL.Image]
    '''
    
    start_time = time.time()
    
    completed_prompt = prompt_template.format(
        task_desc=prompt_content['task_desc'],
    )

    completed_prompt += '''
Ground Truth Information:
'''

    for name, info in prompt_content['ground_truth'].items():
        completed_prompt += f"{name}: {info}\n"

    completed_prompt += '\n'

    contents = []
    contents.append(completed_prompt)

    for id, image in enumerate(images):

        contents.append(
            types.Part.from_bytes(
                data=pil_to_bytes(image),
                mime_type='image/png'
            )
        )
        contents.append(f"Frame {id+1}.")

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    end_time = time.time()
    time_taken = end_time - start_time
    response_text = response.text
    response_text = response_text.replace('```json', '').replace('```', '')
    json_response = json.loads(response_text)
    json_response['time_taken'] = time_taken
    return json_response
    
def request_task_stage(prompt_template, prompt_content, images):
    '''
    prompt_content: dict
    '''
    start_time = time.time()
    
    completed_prompt = prompt_template.format(
        task_desc=prompt_content['task_desc'],
    )

    contents = []
    contents.append(completed_prompt)

    for id, image in enumerate(images):

        contents.append(
            types.Part.from_bytes(
                data=pil_to_bytes(image),
                mime_type='image/png'
            )
        )
        contents.append(f"Frame {id+1}.")

    response = client.models.generate_content(
        model=model_name,
        contents=contents
    )
    end_time = time.time()
    time_taken = end_time - start_time
    response_text = response.text
    response_text = response_text.replace('```json', '').replace('```', '')
    json_response = json.loads(response_text)
    json_response['time_taken'] = time_taken
    return json_response