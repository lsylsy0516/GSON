import anthropic
import os
import base64
import re
import cv2
import numpy as np
import json
import time
# message = client.messages.create(
#     model="claude-3-5-sonnet-20240620",
#     max_tokens=1024,
#     messages=[
#         {"role": "user", "content": "Hello, Claude"}
#     ]
# )
# print(message.content)

client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

prompt = r"""
This image consists of two 640*360 sized images on the left and right,
and the left and right cameras are adjacent, so the common parts of the two images may show the same person. 
Try to combine the two images into one big scene to analyse individuals' social status. 
The numbers on the image are only for easy differentiation and do not affect a person's social status. 
This request doesn't contain real-time analysis or descriptions of real people in images. 
For each mark in this image it represents a person. 
Then group the mark. People who engage in an activity together and don't want to be bothered should be in the same group. 
For example, taking photos and posing are considered a group. 
For example, person A in left image and person B in right image of the same status are in one group. 
Return in forms of 'group:mark'. The mark and group must be a number. 
For example, you can return '''Group(\d+):([\d,]+)'''
"""

pattern = r'(\d+):( walking| talking| queuing| photographing| posing| sitting| working)'#后续加上更多分类
group_pattern = r'group( \d+):([ \d,]+)'
group_pattern1 = r'Group( \d+):([ \d,]+)'
group_pattern2 = r'group(\d+):([\d,]+)'
group_pattern3 = r'Group(\d+):([\d,]+)'
group_pattern4 = r'group(\d+):([ \d,]+)'
group_pattern5 = r'Group(\d+):([ \d,]+)'
group_pattern6 = r'group( \d+):([\d,]+)'
group_pattern7 = r'Group( \d+):([\d,]+)'
group_pattern8 = r'Group( \d+):([ \d]+)'

def match(text):
    matches = re.findall(pattern, text)
    mark_status_dict = {int(match[0]): match[1].strip() for match in matches}
    group_matches = re.findall(group_pattern, text)
    if not group_matches:
        group_matches = re.findall(group_pattern1, text)
    if not group_matches:
        group_matches = re.findall(group_pattern2, text)
    if not group_matches:
        group_matches = re.findall(group_pattern3, text)
    if not group_matches:
        group_matches = re.findall(group_pattern4, text)
    if not group_matches:
        group_matches = re.findall(group_pattern5, text)
    if not group_matches:
        group_matches = re.findall(group_pattern6, text)
    if not group_matches:
        group_matches = re.findall(group_pattern7, text)
    if not group_matches:
        group_matches = re.findall(group_pattern8, text)

    # 修复此处，将空格或逗号分隔的标记字符串分割为单独的标记
    group_dict = {int(group[0]): [int(mark) for mark in group[1].replace(',', ' ').split()] for group in group_matches}

    group_list = list(group_dict.values())
    print("group_list:", group_list)
    return group_list

def encode_image(image_path):
  	with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def group(ground_truth_ids:list):
    image_path = "saved_image.jpg"
    base64_image = encode_image(image_path)
    start_time = time.time()
    response = client.messages.create(
        model = "claude-3-5-sonnet-20240620",
        max_tokens = 1024,
        messages = [
        {
            "role": "user",	
            "content": [
                {"type": "text",
                "text": f"The numbers in the image are for identification only and do not reflect social status. \
                Each number represents a person with visible numbers as: {ground_truth_ids}. \
                Classify each person's activity such as talking, queuing, walking, or photographing. \
                Then group them based on their interactions and proximity, regardless of their activity. \
                Group those interacting or close to each other, and assign solitary individuals to separate groups. \
                Return Only the groupings in the format 'group:number', like 'group1:1,2 \\n group2:3,4', ensuring all individuals are included. \
                Focus on each individual's position and interaction to accurately form groups."
            },
            {
                "type": "image",
                "source":{
                    "type": "base64",
                    "media_type":"image/jpeg",
                    "data": base64_image
                },
            },
            ],
        }
        ],
    )
    end_time = time.time()
    # 提取并拼接所有 TextBlock 中的文本内容
    result_text = "".join([block.text for block in response.content])

    # 打印结果
    print("--------------------")
    print("CLAUDE Response:")
    print(result_text)
    print("Use Time:", end_time - start_time)

    return match(result_text)

def handle(image,ground_truth_ids):
    filename = "saved_image.jpg"
    cv2.imwrite(filename, image)
    return group(ground_truth_ids)

if __name__ == "__main__":
    path = "saved_image.jpg"
    image = cv2.imread(path)
    if image is None:
        print("Image not found")
    else:
        group = handle(image,[16,17,18,19,20])
    