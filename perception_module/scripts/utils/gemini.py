import os
import re
#from google.colab import userdata
import google.generativeai as genai
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import time

text1 = """
0: working
1: working
2: walking

group1: 0,1
group2: 2
"""


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


#gemini-1.5-pro-latest
#gemini-pro-vision
API_KEY = os.environ['GEMINI_API_KEY']
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
genai.configure(api_key=API_KEY)
# img.show()
model = genai.GenerativeModel('gemini-1.5-flash-latest')

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


def test():
    img = PIL.Image.open("./1.jpg")
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    print(response.text)
    match(response.text)

def handle(image):
    filename = "saved_image.jpg"
    cv2.imwrite(filename, image)
    img = PIL.Image.open(filename)
    response = model.generate_content([prompt, img], stream=True)
    response.resolve()
    print(response.text)
    group = match(response.text)

    return group

def group(image_path):
    base64_image = PIL.Image.open(image_path)
    start_time = time.time()
    response = model.generate_content([prompt, base64_image], stream=True)
    response.resolve()
    end_time = time.time()
    print(response.text)
    group = match(response.text)
    print(end_time-start_time)
    return group

if __name__ == '__main__':
  img = cv2.imread("saved_image.jpg")
  handle(img)        

