import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_binary_image(file_path, image_shape=(256, 256)):
    # Đọc dữ liệu từ file nhị phân
    img_data = np.fromfile(file_path, dtype=np.uint8)

    # Reshape dữ liệu thành kích thước hình ảnh mong muốn
    img_data = img_data[:np.prod(image_shape)].reshape(image_shape)

    return img_data.astype(float)

def dft_filter(image, filter_size):
    # Tạo bộ lọc DFT
    filter_kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)
    filter_dft = np.fft.fft2(filter_kernel, s=image.shape)

    # Tính DFT của ảnh và nhân pointwise
    image_dft = np.fft.fft2(image)
    result_dft = image_dft * filter_dft

    # Lấy IDFT để có ảnh kết quả
    result_image = np.fft.ifft2(result_dft).real
    return result_image

# Đọc ảnh và hiển thị
image_path = "salesman.bin"  # Điều chỉnh đường dẫn tới file salesman.bin
original_image = read_binary_image(image_path)

# Áp dụng bộ lọc và hiển thị
dft_filter_image = dft_filter(original_image, 7)

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(dft_filter_image, cmap='gray')
plt.title("Dft filter image")
plt.axis('off')

plt.show()
