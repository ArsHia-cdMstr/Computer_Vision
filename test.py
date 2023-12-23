import numpy as np
import cv2

def add_poisson_noise(image, scale):
    img_float = np.float32(image) / 255.0
    noise = np.random.poisson(img_float * scale) / scale
    noise = noise * 255
    noise = noise.astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

image = cv2.imread('DS/Med/1.png', cv2.IMREAD_GRAYSCALE)
noisy_image = add_poisson_noise(image, scale=2.0)

cv2.imshow('Noisy Image', noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
