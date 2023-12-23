import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def salt_pepper_noise(image, density):
    """
    input:
    - a gray scale image of size (Height, Width)
    - density of noise on image

    output: a gray scale image with salt and pepper noise with size (Height, Width)
    """
    salt_prob = density
    pepper_prob = density
    output = np.zeros(image.shape, np.uint8)
    threshold = 1 - salt_prob
    threshold_2 = 1 - salt_prob - pepper_prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = np.random.random()
            if rdn < pepper_prob:
                output[i][j] = 0
            elif rdn < threshold_2:
                output[i][j] = image[i][j]
            else:
                output[i][j] = 255
    return output


def read_images_recursively(directory):
    """
    input : the parent directory of the images

    directories
    DS :
        - Med
        - Normal
        - RS

    output : a list which 1st Dimension is folder_count, and then it filled gray scale images
        EX :
            [
                Med = 0 : [ image_0 , image_1 , image_2, .... ],
                Normal = 1: [ image_0 , image_1, .....]],
                RS = 2: [ image_0, .....]
            ]
    """
    images_dict = []
    folder_count = 0
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            images_dict.append( [] )  # ایجاد یک لیست برای نگهداری تصاویر هر فولدر

            for file in os.listdir(dir_path):
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # فرمت‌های تصویری که می‌خواهید بخوانید
                    file_path = os.path.join(dir_path, file)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    images_dict[folder_count].append(image)

            folder_count += 1
    return images_dict

image_list = read_images_recursively('DS')

# def use_all_noise(image_dic: dict[str, list], densities: tuple[float]):
#     image_dic['Med']
#     densities
#
