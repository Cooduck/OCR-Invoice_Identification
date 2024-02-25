import os
import numpy as np
import json
from PIL import Image

Image_path = './imagedata/image'
Xml_path = './imagedata/xml'

def crop_image(image_path, point1, point2, name, label):
    # Open the image
    image = Image.open(image_path)

    # Define the coordinates of the two points
    x1, y1 = point1
    x2, y2 = point2

    if x1 > x2:
        tmp = x1
        x1 = x2
        x2 = tmp
        tmp = y1
        y1 = y2
        y2 = tmp

    # Crop the image using the coordinates of the two points
    cropped_image = image.crop((x1, y1, x2, y2))

    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the cropped image with the specified name in the 'data' directory
    if cropped_image.getbbox() and label != '###' and label != '#':
        # 如果不为空，则保存图像
        cropped_image.save('data/{}.jpg'.format(name))
        return True
    else:
        return False

'''
从json文件中读取图像中的真值框
'''
def read_json(file_path):
    with open(file_path, 'r', encoding= 'gbk') as f:
        data = json.load(f)
    return data

def readxml(path):
    gtboxes = []
    labelboxs = []
    data = read_json(path)

    # 获取形状列表
    shapes = data['shapes']

    # 遍历形状列表并提取框的坐标
    for shape in shapes:
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1 = points[0]
            x2, y2 = points[1]
            gtboxes.append((x1, y1, x2, y2))
            labelboxs.append(shape['label'])

    return np.array(gtboxes), np.array(labelboxs)


def traverse_files(folder_path):
    # Iterate over all files in the specified folder
    cnt = 0
    Labelbox = []
    for root, dirs, imgs in os.walk(folder_path):
        for img_name in imgs:
            img_path = os.path.join(root, img_name)
            xml_path = os.path.join(Xml_path, img_name.replace('.jpg', '.json'))

            gtboxes, labelboxs = readxml(xml_path)

            for i in range(len(gtboxes)):
                point1 = [gtboxes[i][0], gtboxes[i][1]]
                point2 = [gtboxes[i][2], gtboxes[i][3]]
                name = 'image' + str(cnt)
                if crop_image(img_path, point1, point2, name, labelboxs[i]):
                    Labelbox.append(labelboxs[i])
                    cnt = cnt + 1
    return Labelbox

Labelbox = []
Labelbox = traverse_files(Image_path)

# 打开（或创建）一个文本文件，以写入模式打开
#with open('information.txt', 'w') as f:
#    for index, label in enumerate(Labelbox):
#        line = './data/image{}.jpg\\t{}'.format(str(index), label)
#        f.write(line + '\n')
