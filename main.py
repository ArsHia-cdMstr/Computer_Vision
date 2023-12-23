import cv2
import numpy as np
import matplotlib.pyplot as plt
import os




def read_images_recursively(directory):
    """
    input : the parent directory of the images
    :param directory:
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
            images_dict.append([])  # ایجاد یک لیست برای نگهداری تصاویر هر فولدر

            for file in os.listdir(dir_path):
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # فرمت‌های تصویری که می‌خواهید بخوانید
                    file_path = os.path.join(dir_path, file)
                    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    images_dict[folder_count].append(image)

            folder_count += 1
    return images_dict


image_list = read_images_recursively('DS')


def salt_pepper_noise(image, density):
    density /= 5
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


def speckle_noise(image, variance):
    # ایجاد نویز speckle
    noise = np.random.normal(0, variance, image.shape)
    speckle_noisy = image + image * noise
    speckle_noisy = np.clip(speckle_noisy, 0, 255)

    return speckle_noisy


def add_poisson_noise(image, scale):
    img_float = np.float32(image) / 255.0
    noise = np.random.poisson(img_float * scale) / scale
    noise = noise * 255
    noise = noise.astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image


def choose_noise_and_implement(noise_name, image):
    """
    :param noise_name:
    :param image:

    we only use this function to show
    you some examples of implementation
    of noises on images
    """
    if noise_name == "salt_and_pepper":
        return salt_pepper_noise(image, 0.2)
    if noise_name == "speckle":
        return speckle_noise(image, 0.002)
    if noise_name == "Possion":
        return add_poisson_noise(image, 1.0)
    if noise_name == "Gaussian":
        pass


def store_noisy_images(image_list: list):
    """
    Store noisy images according to :
    - densities
    - the variances
    - the noise scaling
    :param image_list:
    """
    folder_names = {
        0: "MED_noisy",
        1: "Normal_noisy",
        2: "RS_noisy",
    }

    noises = ["salt_and_pepper", "speckle", "Possion", "Gaussian"]

    densities = (0.05, 0.1, 0.2, 0.4)
    variances = (0.0001, 0.0005, 0.001, 0.002)
    noise_scaling = (0.1, 0.5, 1.0, 2.0)

    for noise_name in noises:
        for folder_id in range(len(image_list)):
            image_counter = 1
            for image in image_list[folder_id]:

                noisy_image = choose_noise_and_implement(noise_name, image)
                path = f"denoised_images/{folder_names[folder_id]}/{noise_name}"
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(f"{path}/{image_counter}.png", noisy_image)
                image_counter += 1


store_noisy_images(image_list)
