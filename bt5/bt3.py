import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_binary_image(file_path, image_shape=(256, 256)):
    # Đọc dữ liệu từ file nhị phân
    img_data = np.fromfile(file_path, dtype=np.uint8)

    # Reshape dữ liệu thành kích thước hình ảnh mong muốn
    img_data = img_data[:np.prod(image_shape)].reshape(image_shape)

    return img_data.astype(float)

def linear_average_filter(image, filter_size):
    # Tạo bộ lọc trung bình tuyến tính
    filter_kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)

    # Áp dụng convolution trong không gian ảnh
    result_image = cv2.filter2D(image, -1, filter_kernel, borderType=cv2.BORDER_CONSTANT)
    return result_image

def zero_phase_dft_filter(image, filter_size):
    # Tạo zero-phase impulse response
    filter_kernel = np.ones((filter_size, filter_size), dtype=np.float32) / (filter_size**2)
    filter_kernel = np.pad(filter_kernel, ((0, image.shape[0] - filter_size), (0, image.shape[1] - filter_size)), mode='constant')

    # Tính DFT của ảnh và nhân pointwise
    image_dft = np.fft.fft2(image)
    result_dft = image_dft * np.fft.fft2(filter_kernel)

    # Lấy IDFT để có ảnh kết quả
    result_image = np.fft.ifft2(result_dft).real
    return result_image

# Đọc ảnh và hiển thị
image_path = "salesman.bin"  # Điều chỉnh đường dẫn tới file salesman.bin
original_image = read_binary_image(image_path)

# (a) Áp dụng convolution và hiển thị
convolution_filter_image = linear_average_filter(original_image, 7)

# (b) Áp dụng zero-phase DFT filter và hiển thị
zero_phase_dft_filter_image = zero_phase_dft_filter(original_image, 7)

# Hiển thị kết quả so sánh
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(convolution_filter_image, cmap='gray')
plt.title("Convolution filter image")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(zero_phase_dft_filter_image, cmap='gray')
plt.title("Zero phase DFT filter image")
plt.axis('off')

plt.show()
