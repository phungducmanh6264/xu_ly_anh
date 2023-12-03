import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Tạo ma trận COLS và ROWS
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

# Định nghĩa hình ảnh I1
u0, v0 = 2, 2
I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (2.0 * COLS + 2.0 * ROWS))
I2 = 0.5 * np.exp(-1j * 2 * np.pi / 8 * (2.0 * COLS + 2.0 * ROWS))
I4 = -1j * (I1 - I2)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.real(I4), cmap='gray')
plt.title('Real I4')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.imag(I4), cmap='gray')
plt.title('Imag I4')
plt.axis('off')

Itilde4 = fftshift(fft2(I4))

plt.subplot(2, 2, 3)
plt.imshow(np.real(Itilde4), cmap='gray')
plt.title('Real Itilde4')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.imag(Itilde4), cmap='gray')
plt.title('Imag Itilde4')
plt.axis('off')

plt.show()
