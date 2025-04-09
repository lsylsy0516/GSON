input_list = "1 2,3"

# 按逗号分隔字符串
parts = input_list.split(',')

# 对每个部分按空格分隔，并将字符串转换为整数
result = [[int(num) for num in part.split()] for part in parts]

print(result)