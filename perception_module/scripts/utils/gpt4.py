import base64
import re
import cv2
import numpy as np
import json
import os
import time
from openai import OpenAI



pattern = r'(\d+):( walking| talking| queuing| photographing| posing| sitting| working)'#后续加上更多分类
group_pattern = r'group( \d+):([ \d,]+)'
group_pattern1 = r'Group( \d+):([ \d,]+)'
group_pattern2 = r'group(\d+):([\d,]+)'
group_pattern3 = r'Group(\d+):([\d,]+)'
group_pattern4 = r'group(\d+):([ \d,]+)'
group_pattern5 = r'Group(\d+):([ \d,]+)'
group_pattern6 = r'group( \d+):([\d,]+)'
group_pattern7 = r'Group( \d+):([\d,]+)'
group_pattern8 = r'Group( \d+):([ \d,]+)'
group_pattern9 = r'group((\d+)):([\d,]+)'
group_pattern10 = r'Group((\d+)):([\d,]+)'
group_pattern11 = r'group((\d+)):([ \d,]+)'
group_pattern12 = r'Group((\d+)):([ \d,]+)'
group_pattern13 = r'group(\d+):([\d, ]+)'
group_pattern14 = r'Group(\d+):([\d, ]+)'
group_pattern15 = r'Group( \d+):([\d, ]+)'
group_pattern16 = r'group( \d+):([\d, ]+)'
# MODEL = "gpt-4-vision-preview"
MODEL = "gpt-4o"


# You should set your OpenAI API key here
# os.environ['OPENAI_API_KEY'] = 


#For each mark in this image it represents a person. Judge each mark what he is doing, talking or queuing or other social status. Return in forms of 'mark:status'. The mark must be a number and the status must be talking, queuing, walking or photographing. Then group the mark. For the person who engage in the same activity they should be in the same group. Return in forms of 'group:mark'. The mark and group must be a number.
def encode_image(image_path):
  	with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
  

def match(text):
    result = []
    for line in text.strip().split('\n'):
        # 拆分 group 和 numbers 部分
        group, numbers = line.split(':')
        # 将 numbers 转换为整数列表
        number_list = [int(num.strip()) for num in numbers.split(',')]
        # 将结果添加到二维列表中
        result.append(number_list)
    print("group_list:", result)
    return result
    

def group(image_path,ground_truth_ids:list):
	# image_path = "/home/orin/planner_ws/src/perception_module/scripts/utils/saved_image.jpg"
	base64_image = encode_image(image_path)
	client = OpenAI()
	start_time = time.time()
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
		{
			"role": "user",	
			"content": [
				{"type": "text",
				"text": f"Task: Assume you are a social robot and you need to avoid the crowds in the picture and not disturb the same group of people. Please group the people in the picture accordingly.\
							Visual Prompting: The numbers in the image are for identification only and do not reflect social status. Each number represents a person with visible numbers as: {ground_truth_ids}. \
							Remember: Group those interacting or close to each other. Ensure all individuals are included. Focus on each individual's position and interaction to accurately form groups.\
								Pay more attention to everyone's orientation and interactions to ensure that as many people as possible are grouped together. Anyone doing the same thing should be grouped together as one group. \
								For example, taking photos and those posing for photos are the same. People who will influence each other should be grouped together. Even if some people are far away or people do different activities, \
								as long as they interact, they should be grouped together. Each number should be grouped only once. This image consists of two 640*360 sized images on the left and right, \
								and the left and right cameras are adjacent, so the common parts of the two images may show the same person. \
								Try to combine the two images into one big scene to analyse individuals' social status. The people in this image are more likely to be in one group.\
							Answer Format:Return Only the groups in the format 'group:number', like 'group1:1,2 \n group2:3,4'. One line for one group."
			},
			{
				"type": "image_url",
				"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}",
				},
			},
			],
		}
		],
		max_tokens=500,
	)
	end_time = time.time()
	print("--------------------")
	print("GPT4V-preview Response:")
	print(response.choices[0].message.content)
	print("Use Time:", end_time - start_time)
	txt_path = image_path.replace(".jpg", ".txt")
	with open(txt_path, 'w') as txt_file:
		txt_file.write(response.choices[0].message.content)
    
	print(f"Response content saved to: {txt_path}")
	return match(response.choices[0].message.content)

def group_debug(ground_truth_ids:list):
	image_path = "/home/orin/planner_ws/src/perception_module/keyframes/2024-09-11_213322/3405.jpg"
	base64_image = encode_image(image_path)
	client = OpenAI()
	start_time = time.time()
	response = client.chat.completions.create(
		model=MODEL,
		messages=[
		{
			"role": "user",	
			"content": [
				{"type": "text",
				"text": f"The numbers in the image are for identification only and do not reflect social status. \
				Each number represents a person with visible numbers as: {ground_truth_ids}. \
				Classify each person's activity such as talking, queuing, walking, or photographing. \
				Then group them based on their interactions and proximity, regardless of their activity. \
				Group those interacting or close to each other, and assign solitary individuals to separate groups. \
				Return Only the groupings in the format 'group:number' and the reason why you group like that, like 'group1:1,2 \\n group2:3,4', ensuring all individuals are included. \
				Focus on each individual's position and interaction to accurately form groups."
			},
			{
				"type": "image_url",
				"image_url": {
				"url": f"data:image/jpeg;base64,{base64_image}",
				},
			},
			],
		}
		],
		max_tokens=500,
	)
	end_time = time.time()
	print("--------------------")
	print(f"{MODEL} Response:")
	print(response.choices[0].message.content)
	print("Use Time:", end_time - start_time)
	return match(response.choices[0].message.content)
		
def handle(image,ground_truth_ids):
    filename = "/home/orin/planner_ws/src/perception_module/keyframes/2024-09-11_213322/3405.jpg"  
    cv2.imwrite(filename, image)
    return group(filename,ground_truth_ids)
	
        

if __name__ == '__main__':
	try:
		# path = input("Please input the image path:")
		path = "/home/orin/planner_ws/src/perception_module/keyframes/2024-09-11_215924/846.jpg"
		# path = "/home/orin/planner_ws/src/perception_module/keyframes/2024-09-07_182135445.jpg"  
		image = cv2.imread(path)
		if image is None:
			raise Exception("Image not found")
		else:
			ground_truth_ids = input("Please input the ground truth ids:")
			ground_truth_ids = list(map(int,ground_truth_ids.split(',')))
		group_list = handle(image,ground_truth_ids)
	except Exception as e:
		print(e)
		group_list = []

