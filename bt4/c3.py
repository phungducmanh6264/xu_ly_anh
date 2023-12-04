import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt

# Tạo ma trận COLS và ROWS
COLS, ROWS = np.meshgrid(np.arange(8), np.arange(8))

# Định nghĩa hình ảnh I1
u0, v0 = 2, 2
I3 = np.cos(2 * np.pi * (u0 * COLS + v0 * ROWS))

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(np.real(I3), cmap='gray')
plt.title('Real I3')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.imag(I3), cmap='gray')
plt.title('Imag I3')
plt.axis('off')

Itilde3 = fftshift(fft2(I3))

plt.subplot(2, 2, 3)
plt.imshow(np.real(Itilde3), cmap='gray')
plt.title('Real Itilde3')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(np.imag(Itilde3), cmap='gray')
plt.title('Imag Itilde3')
plt.axis('off')

plt.show()