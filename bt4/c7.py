from PIL import Image
import numpy as np
from scipy.fft import fftshift, fft2
import matplotlib.pyplot as plt


def read_binary_image(file_path, image_shape=(256, 256)):
    # Đọc dữ liệu từ file nhị phân
    img_data = np.fromfile(file_path, dtype=np.uint8)

    # Reshape dữ liệu thành kích thước hình ảnh mong muốn
    img_data = img_data[:np.prod(image_shape)].reshape(image_shape)

    return img_data.astype(float)

# Đọc hình ảnh từ file camera.bin
I6 = read_binary_image('camera.bin').astype(float)

# Định nghĩa hình ảnh J1 và J2
J1 = np.abs(I6)
J2 = np.angle(I6)

# Hiển thị J2
plt.subplot(1, 2, 1)
plt.imshow(J2, cmap='gray')
plt.title('J2: Phase of Original Image')
plt.axis('off')

# Xử lý hiển thị J1 với log để tăng độ tương phản
JJ1 = np.log(J1 + 1)
plt.subplot(1, 2, 2)
plt.imshow(JJ1, cmap='gray')
plt.title('JJ1: Log of Magnitude of Original Image')
plt.axis('off')

plt.show()
