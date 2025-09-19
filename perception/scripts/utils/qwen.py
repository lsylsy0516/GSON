from openai import OpenAI
from PIL import Image
from io import BytesIO
import base64
import requests
import re
import cv2
import numpy as np
import json
import os
import time

client = OpenAI(
    base_url="https://adacomp.ngrok.app/v1",
)

MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

def match(text, ground_truth_ids):
    result = []
    used_ids = set()
    text = text.replace("`", "")
    for line in text.strip().split('\n'):
        if not line.strip():
            continue
        try:
            if ':' in line:
                _, rest = line.split(':', 1)
                ids_str = rest.strip().split(' ')[0]  # Extract ID list part (truncate at explanation)
                raw_ids = [x.strip() for x in ids_str.split(',') if x.strip().isdigit()]
                id_list = []
                for x in raw_ids:
                    idx = int(x)
                    if idx in ground_truth_ids and idx not in used_ids:
                        id_list.append(idx)
                        used_ids.add(idx)
                if id_list:  # Add only if not empty
                    result.append(id_list)
        except Exception as e:
            print(f"Failed to parse line: {line}\nError: {e}")
    print("Filtered group_list:", result)
    return result

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def group_with_status(image_path, ground_truth_ids, person_status_dict):
    """
    Group people based on their status and ground truth IDs.
    """
    base64_image = encode_image(image_path)
    start_time = time.time()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
        {"role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"""
            Task: You are a social robot that needs to avoid crowds and ensure you do not disturb the same group of people.

            Grouping Rules:
            - Each number in the image represents one person (e.g., {ground_truth_ids}). Numbers are for identification only.
            - Your goal is to group people who are interacting or socially connected, even if the group contains only one individual.
            - Use **visual cues from the image** (such as body orientation, mutual gaze, physical distance, shared gestures, and positioning) as the **primary evidence** for grouping.
            - The listed "activity" of each person is only a **secondary hint**, and should **not override what is visible in the image**. Do **not assume two people are grouped just because they share the same activity**.
            - People doing the same activity (like walking or sitting) but far apart with no interaction should not be grouped together.
            - People performing different activities can be in the same group if visual cues (gaze, body direction, gesture, physical proximity) suggest interaction.
            - If a person's activity is marked as "Unknown", you must rely entirely on the image to infer their possible activity and social interaction. Do not ignore them. They must still be grouped based on visual cues. 
            - **Only include and include ALL** the given IDs ({ground_truth_ids}) in your response. Do not introduce new IDs or repeat them.

            Answer Format (Return only the groups in the following format):
            group1: 0,1 explanation...
            group2: 2,3,4 explanation...


            Each group should be on a separate line with brief reasoning based on image evidence (e.g., "facing each other", "taking photo together", "close and aligned").
            """
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Here is a picture of the robot's view. Try grouping those pedestrians with the given IDs: {ground_truth_ids}.

            Each person's activity:
            """ + '\n'.join([f"ID {pid}: {activity}" for pid, activity in person_status_dict.items()])
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
            ]
        }
        ],
        max_tokens=500,
        temperature=0.1,
    )
    end_time = time.time()
    print("--------------------")
    print("Response:")
    print(response.choices[0].message.content)
    print(f"start_time: {start_time}, end_time: {end_time}")
    print("Use Time:", end_time - start_time)
    return match(response.choices[0].message.content, ground_truth_ids)

def group_naive(image_path, ground_truth_ids):
    """
    Naive grouping based on ground truth IDs.
    """
    base64_image = encode_image(image_path)
    start_time = time.time()
    response = client.chat.completions.create(
            model=MODEL,
            messages=[
            {
                "role": "system",    
                "content": [
                    {"type": "text",
                    "text": f"Task: You are a social robot that needs to avoid crowds and ensure you do not disturb the same group of people.  \
                        Grouping Rules:  \
                        - The numbers in the image are for identification only and do not reflect social status. Each number represents a visible person: {ground_truth_ids}. \
                        - Pay close attention to each person's body orientation, facial direction, and interactions to accurately form groups.  \
                        - Group people who are interacting with each other, ensuring all individuals are included in groups,even single is accepted.  \
                        - People engaged in the same activity should be grouped together (e.g., those taking photos and those posing for photos even not looking to others belong to the same group).  \
                        - Even if individuals are far apart or performing different activities, they should be grouped together if they are interacting.  \
                        - Only include the given IDs in the response. Do not introduce any extra or missing IDs and do not repeat same ID.  \
                        Answer Format (Return only the groups in the following format):  \
                        ```\n \
                        group1:1,2,3 \n \
                        group2:4,5 \n \
                        ```  \
                        Each group should be on a separate line, with no extra explanations."
                },

                ],
            },
            {
                "role": "user",
                "content":[
                {
                    "type": "text",
                    "text": "Here is a picture of robot view,try group those pedestrains ."
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                }
                ]
            }
            ],
            max_tokens=500,
            temperature=0.1,
        )

    end_time = time.time()
    print("--------------------")
    print("Use Time:", end_time - start_time)
    return match(response.choices[0].message.content, ground_truth_ids)
