import os
from openai import AzureOpenAI
import base64

from dotenv import load_dotenv
load_dotenv()


model = 'gpt-4o'
api_key = os.getenv("API_KEY")

client = AzureOpenAI(  
    azure_endpoint="https://ai-gaoyang0921ai351861530836.openai.azure.com",  
    api_key=api_key,
    api_version="2024-12-01-preview",
)

frame_fix_prompt = '''
Target:
You are a professional error detector (image QA). Task: analyze the provided frame and task description. Given the frame at that timestamp, the robot will do a motion primitive action to continue the task. You need to determine whether the motion primitive action is correct to help complete the task; if not correct, provide the failure reason and then provide the correct action primitives that, if executed correctly, would have led to successful task completion.


Analysis requirements:
1. Watch the frame carefully and the given task description.
2. Definition of "Completed": only when **all** core actions/results specified in the task_description are visibly and verifiably executed in the video should you mark it as Completed.
3. If visual occlusion or missing key details prevents verification of a required step, mark as Not Completed and specify which evidence is missing.
4. If the action primitive provided is not correct, provide the failure reason from the list given below.
5. If you base your judgment on any assumption, state that assumption briefly in the Evidence Note (one sentence).
6. If you think the action primitive provided is not correct, provide the correct action primitives that, if executed correctly, would have led to successful task completion. The action primitives should be selected from the list given below. After executing the primitive, the root failure reason should be resolved.
7. Your output must **strictly follow the Return Format below** and **only output** four lines (field name and value). Do not add any additional commentary.

Failure Reasons (choose one):
1. Did Not Move Base Cube Close Enough
2. Placed Top Cube in Unstable Position
3. Use the wrong color cube as the top cube

Action Primitives (choose from the list below):
1. Move Left
2. Move Right
3. Move Forward
4. Move Backward
5. Move Up
6. Move Down
7. Close Gripper
8. Open Gripper

Return Format (must match exactly):
Completed: [yes/no]
failure reason: [If completed write "none"; otherwise choose a failure reason]
evidence: [If completed write "none"; otherwise one-sentence explanation of the key evidence at that timestamp/interval (include any assumption), ¡Ü30 words]
fix primitive: [If completed write "none"; otherwise a single action primitives that would have led to successful task completion, e.g., "Move Left"]

Notes:
- If the task contains multiple subtasks, only consider it Completed if all subtasks are verified completed.

Task Description:
{task_desc}

Proposed action primitive the robot will do: 
{proposed_primitive}
'''

def encode_image_by_path(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def encode_image_by_pil(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



def request_openai(prompt_content, images, model=model):
    '''
    prompt_content: dict
    images: List[PIL.Image]
    '''
    completed_prompt = frame_fix_prompt.format(
        task_desc=prompt_content['task_desc'],
        proposed_primitive=prompt_content['proposed_primitive'],
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": completed_prompt},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{encode_image_by_pil(images[0])}"}
                    }
                ]
            },

        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def analyze_openai_response(response):
    lines = response.split('\n')
    result = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result