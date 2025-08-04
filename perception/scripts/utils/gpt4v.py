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

def match(text):
	result = []
	text = text.replace("`","")
	for line in text.strip().split('\n'):
		# 拆分 group 和 numbers 部分
		group, numbers = line.split(':')
		# 将 numbers 转换为整数列表
		number_list = [int(num.strip()) for num in numbers.split(',')]
		# 将结果添加到二维列表中
		result.append(number_list)
	print("group_list:", result)
	return result

    
# MODEL = "chatgpt-4o-latest"
MODEL = "gpt-4.5-preview-2025-02-27"
# MODEL = "o1-2024-12-17"
# MODEL = "gpt-4o"
flag = 1

# You should set your OpenAI API key here
# os.environ['OPENAI_API_KEY'] = 


#For each mark in this image it represents a person. Judge each mark what he is doing, talking or queuing or other social status. Return in forms of 'mark:status'. The mark must be a number and the status must be talking, queuing, walking or photographing. Then group the mark. For the person who engage in the same activity they should be in the same group. Return in forms of 'group:mark'. The mark and group must be a number.
def encode_image(image_path):
  	with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
  

# def match(text):
# 	matches = re.findall(pattern, text)
# 	mark_status_dict = {int(match[0]): match[1] for match in matches}

# 	group_matches = re.findall(group_pattern, text)
# 	group_dict = {int(group[0]): [int(mark) for mark in group[1].split(',')] for group in group_matches}
# 	group_list = list(group_dict.values())
# 	# print(mark_status_dict)
# 	print("result:")
# 	print(group_list)
# 	return group_list
    
# "The numbers in the image are for identification only and do not reflect social status. \
# 				Each number represents a person with visible numbers as: {ground_truth_ids}. \
# 				Classify each person's activity such as talking, queuing, walking, or photographing. \
# 				Then group them based on their interactions and proximity, regardless of their activity. \
# 				Group those interacting or close to each other, and assign solitary individuals to separate groups. \
# 				Return Only the groupings in the format 'group:number', like 'group1:1,2 \\n group2:3,4', ensuring all individuals are included. \
# 				Focus on each individual's position and interaction to accurately form groups."
def group(image_path,ground_truth_ids:list):
	# image_path = "saved_image_gpt.jpg"
	base64_image = encode_image(image_path)
	client = OpenAI()
	start_time = time.time()
	if flag:
		response = client.chat.completions.create(
			model=MODEL,
			messages=[
			{#Classify each person's activity such as talking, queuing, walking, or photographing.
				"role": "system",	
				"content": [
					{"type": "text",
					"text": f"Task: You are a social robot that needs to avoid crowds and ensure you do not disturb the same group of people.  \
						Grouping Rules:  \
						- The numbers in the image are for identification only and do not reflect social status. Each number represents a visible person: {ground_truth_ids}. \
						- Group people who are interacting with each other, ensuring all individuals are included in groups,even single is accepted.  \
						- Pay close attention to each person's body orientation, facial direction, and interactions to accurately form groups.  \
      					- People engaged in the same activity should be grouped together (e.g., those taking photos and those posing for photos even not looking to others belong to the same group \
							,however, if a person is running and another person is standing taking photos, they should be in different groups even if they are close to each other).  \
						- Even if individuals are far apart or performing different activities, they should be grouped together if they are interacting.  \
						- Only include the given IDs ({ground_truth_ids}) in the response. Do not introduce any extra or missing IDs and do not repeat same ID.  \
						Answer Format (Return only the groups in the following format):  \
						```\n \
						group1:0,1\n \
						group2:2,3\n \
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
					"text": "Here is a picture of robot view,try group those pedestrains with given id :{ground_truth_ids}."
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
		
	else:
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
			# max_tokens=500,
		)
	end_time = time.time()
	print("--------------------")
	print("GPT4V-preview Response:")
	print(response.choices[0].message.content)
	print("Use Time:", end_time - start_time)
	return match(response.choices[0].message.content)

def group_debug(ground_truth_ids:list):
	image_path = "saved_image_gpt.jpg"
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
		# max_tokens=500,
	)
	end_time = time.time()
	print("--------------------")
	print(f"{MODEL} Response:")
	print(response.choices[0].message.content)
	print("Use Time:", end_time - start_time)
	return match(response.choices[0].message.content)
		
def handle(image,ground_truth_ids):
    filename = "saved_image_gpt.jpg"  
    cv2.imwrite(filename, image)
    return group(filename,ground_truth_ids)
	
        

if __name__ == '__main__':
	try:
		# path = input("Please input the image path:")
		path = "/home/orin/planner_ws/src/perception_module/keyframes/2025-03-17_220314/0.jpg"
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
    
	# text = """group1:1,0 \n group2:2 \n group3:3"""
	# match(text)
	
	# text ="""
# ```
# group1:8,10,9,6  
# group2:11,0,13  
# group3:3,4,2,5,1,12,7  
# ```
# 	"""
# 	text.replace("`","")
# 	print(text.replace("`",""))
# 	match(text.replace("`",""))


