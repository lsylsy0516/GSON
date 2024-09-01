import base64
import re
import cv2
import numpy as np
import json
import os
import time
from openai import OpenAI



pattern = r'(\d+):(not person|walking|talking|queuing|photographing)'#后续加上更多分类
group_pattern = r'group(\d+):([\d,]+)'
MODEL = "gpt-4-vision-preview"


# You should set your OpenAI API key here
# os.environ['OPENAI_API_KEY'] = 


#For each mark in this image it represents a person. Judge each mark what he is doing, talking or queuing or other social status. Return in forms of 'mark:status'. The mark must be a number and the status must be talking, queuing, walking or photographing. Then group the mark. For the person who engage in the same activity they should be in the same group. Return in forms of 'group:mark'. The mark and group must be a number.
def encode_image(image_path):
  	with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
  

def match(text):
	matches = re.findall(pattern, text)
	mark_status_dict = {int(match[0]): match[1] for match in matches}

	group_matches = re.findall(group_pattern, text)
	group_dict = {int(group[0]): [int(mark) for mark in group[1].split(',')] for group in group_matches}
	group_list = list(group_dict.values())
	# print(mark_status_dict)
	print("result:")
	print(group_list)
	return group_list
    

def group(ground_truth_ids:list):
	image_path = "saved_image.jpg"
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
				Return Only the groupings in the format 'group:number', like 'group1:1,2 \\n group2:3,4', ensuring all individuals are included. \
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
	print("GPT4V-preview Response:")
	print(response.choices[0].message.content)
	print("Use Time:", end_time - start_time)
	return match(response.choices[0].message.content)

def group_debug(ground_truth_ids:list):
	image_path = "saved_image.jpg"
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
    filename = "saved_image.jpg"  
    cv2.imwrite(filename, image)
    return group(ground_truth_ids)
	
        

if __name__ == '__main__':
	try:
		# path = input("Please input the image path:")
		path = "saved_image.jpg"
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

