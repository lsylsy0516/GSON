import os
import re

# Provided string
text = """
1:not person
2:not person
3:not person
4:not person
5:walking
6:talking
7:not person
8:not person
9:not person
10:not person
11:talking
12:not person
13:talking

group1:5
group2:6,11,13
"""

pattern = r'(\d+):(not person|walking|talking|queuing|photographing)'
group_pattern = r'group(\d+):([\d,]+)'

def match(text1):
	# Find all matches in the text
	matches = re.findall(pattern, text1)
	# Convert matches to a dictionary {mark: status}
	mark_status_dict = {int(match[0]): match[1] for match in matches}
	
	# Find all matches for groups in the text
	group_matches = re.findall(group_pattern, text1)
	# Convert group matches to a dictionary {group: [marks]}
	group_dict = {int(group[0]): [int(mark) for mark in group[1].split(',')] for group in group_matches}
	
	print(mark_status_dict)
	print(group_dict)
	return group_dict
	#print(mark_status_dict[5])
	#print(group_dict[2])
	
if __name__=='__main__':
	match(text)
