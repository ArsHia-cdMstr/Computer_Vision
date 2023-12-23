import numpy as np
import cv2
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std=0):
    # تصویر را به float32 تبدیل می‌کنیم
    img_float = np.float32(image)
    # نویز گوسی را می‌سازیم
    noise = np.random.normal(mean, std, image.shape)
    noise = noise.astype(np.uint8)
    # نویز را به تصویر اضافه می‌کنیم
    noisy_image = cv2.add(image, noise)
    # تصاویر را به بازه 0 تا 255 محدود می‌کنیم
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image

# تصویر اولیه را بارگیری می‌کنیم
image = cv2.imread('DS/Med/1.png', cv2.IMREAD_GRAYSCALE)

# نویز Gaussian را به تصویر اضافه می‌کنیم
noisy_image = add_gaussian_noise(image, mean=0, std=40)

# تصویر نهایی را نمایش می‌دهیم با استفاده از matplotlib
plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(noisy_image, cmap='gray'), plt.title('Image with Gaussian Noise')
plt.show()
