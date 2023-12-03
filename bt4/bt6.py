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

# Đọc các hình ảnh từ file
images = ['camera.bin', 'salesman.bin', 'head.bin', 'eyeR.bin']

imgsData = []


for image_file in images:
    # Đọc hình ảnh từ file
    img = read_binary_image(image_file).astype(float)
    
    # Tính DFT và căn giữa
    img_dft = fftshift(fft2(img))
    magnitude_spectrum = np.log(np.abs(img_dft) + 1)  # Log để tăng độ tương phản
    phase_spectrum = np.angle(img_dft)

    data = []
    data.append({"img": img, "title": "Original Image"})
    data.append({"img": np.real(img_dft), "title": "Real Part of DFT"})
    data.append({"img": np.imag(img_dft), "title": "Imaginary Part of DFT"})
    data.append({"img": magnitude_spectrum, "title": "DFT Log-Magnitude Spectrum"})
    data.append({"img": phase_spectrum, "title": "DFT Phase Spectrum"})

    imgsData.append(data)


_index = 1

for i in range(4):
    for j in range(5):
        img = imgsData[i][j]['img']
        title = imgsData[i][j]['title']
        print(title)

        plt.subplot(4, 5, _index)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

        _index += 1

plt.show()