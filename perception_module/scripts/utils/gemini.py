import os
import re
#from google.colab import userdata
import google.generativeai as genai
# from google import genai
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import time

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
API_KEY = os.environ['GEMINI_API_KEY']
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro-latest')
# model = genai.GenerativeModel('gemini-1.5-pro-exp-0827')
#gemini-1.5-pro-latest
#gemini-pro-vision
flag = 1

#os.environ['API_KEY'] = userdata.get('API_KEY')
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
group_pattern9 = r'group((\d+)):([\d,]+)'
group_pattern10 = r'Group((\d+)):([\d,]+)'
group_pattern11 = r'group((\d+)):([ \d,]+)'
group_pattern12 = r'Group((\d+)):([ \d,]+)'

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
    if not group_matches:
        group_matches = re.findall(group_pattern9, text)
    if not group_matches:
        group_matches = re.findall(group_pattern10, text)
    if not group_matches:
        group_matches = re.findall(group_pattern11, text)
    if not group_matches:
        group_matches = re.findall(group_pattern12, text)
    # 修复此处，将空格或逗号分隔的标记字符串分割为单独的标记
    group_dict = {int(group[0]): [int(mark) for mark in group[1].replace(',', ' ').split()] for group in group_matches}
    group_list = list(group_dict.values())
    print("group_list:", group_list)
    return group_list

def test():
    img = PIL.Image.open("./1.jpg")
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    print(response.text)
    match(response.text)

def group(ground_truth_ids:list):
    image_path = "saved_image_gemini.jpg"
    base64_image = PIL.Image.open(image_path)
    start_time = time.time()
    if flag:
        prompt = f"Task: You are a social robot that needs to avoid crowds and ensure you do not disturb the same group of people.  \
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
    else:
        prompt = f"Task: Assume you are a social robot and you need to avoid the crowds in the picture and not disturb the same group of people. Please group the people in the picture accordingly.\
							Visual Prompting: The numbers in the image are for identification only and do not reflect social status. Each number represents a person with visible numbers as: {ground_truth_ids}. \
							Remember: Group those interacting or close to each other. Ensure all individuals are included. Focus on each individual's position and interaction to accurately form groups.\
								Pay more attention to everyone's orientation and interactions to ensure that as many people as possible are grouped together. Anyone doing the same thing should be grouped together as one group. \
								For example, taking photos and those posing for photos are the same. People who will influence each other should be grouped together. Even if some people are far away or people do different activities, \
								as long as they interact, they should be grouped together.\
							Answer Format:Return Only the groups in the format 'group:number', like 'group1:1,2 \n group2:3,4'. One line for one group."
    response = model.generate_content([prompt, base64_image], stream=True)
    response.resolve()
    end_time = time.time()
    # 提取并拼接所有 TextBlock 中的文本内容
    result_text = response.text
    result_text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    # 打印结果
    print("--------------------")
    print("GEMINI Response:")
    print(result_text)
    print("Use Time:", end_time - start_time)

    return match(result_text)

def handle(image,ground_truth_ids):
    filename = "saved_image_gemini.jpg"  
    cv2.imwrite(filename, image)
    return group(ground_truth_ids)

if __name__ == '__main__':  
    try:
        path = "/home/orin/planner_ws/src/perception_module/keyframes/2025-03-11_164545/2.jpg"
        print(path)
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
