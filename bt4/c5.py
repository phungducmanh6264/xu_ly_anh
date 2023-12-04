import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Tạo ma trận COLS và ROWS
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

# Định nghĩa hình ảnh I1
u1, v1 = 1.5, 1.5
I5 = np.cos(2 * np.pi * (u1 * COLS + v1 * ROWS))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.real(I5), cmap='gray')
plt.title('Real I5')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.imag(I5), cmap='gray')
plt.title('Imag I5')
plt.axis('off')

Itilde5 = fftshift(fft2(I5))

plt.subplot(2, 2, 3)
plt.imshow(np.real(Itilde5), cmap='gray')
plt.title('Real Itilde5')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.imag(Itilde5), cmap='gray')
plt.title('Imag Itilde5')
plt.axis('off')

plt.show()
