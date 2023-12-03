import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Tạo ma trận COLS và ROWS
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

# Định nghĩa hình ảnh I1
u0, v0 = 2, 2
I1 = 0.5 * np.exp(1j * 2 * np.pi / 8 * (2.0 * COLS + 2.0 * ROWS))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.real(I1), cmap='gray')
plt.title('Real I1')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.imag(I1), cmap='gray')
plt.title('Imag I1')
plt.axis('off')

Itilde1 = fftshift(fft2(I1))

plt.subplot(2, 2, 3)
plt.imshow(np.real(Itilde1), cmap='gray')
plt.title('Real Itilde1')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.imag(Itilde1), cmap='gray')
plt.title('Imag Itilde1')
plt.axis('off')

plt.show()
