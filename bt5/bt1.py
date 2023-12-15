import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_binary_image(file_path, image_shape=(256, 256)):
    # Đọc dữ liệu từ file nhị phân
    img_data = np.fromfile(file_path, dtype=np.uint8)

    # Reshape dữ liệu thành kích thước hình ảnh mong muốn
    img_data = img_data[:np.prod(image_shape)].reshape(image_shape)

    return img_data.astype(float)

def linear_average_filter(image):
    filter_size = 7
    filter_kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)
    result = cv2.filter2D(image, -1, filter_kernel, borderType=cv2.BORDER_CONSTANT)
    return result

# Đọc ảnh và hiển thị
image_path = "salesman.bin"  # Điều chỉnh đường dẫn tới file salesman.bin
original_image = read_binary_image(image_path)

# Áp dụng bộ lọc và hiển thị
filtered_image_a = linear_average_filter(original_image)

plt.subplot(1, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image_a, cmap='gray')
plt.title("Filtered image")
plt.axis('off')

plt.show()
