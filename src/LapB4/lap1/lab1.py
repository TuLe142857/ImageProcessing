import scipy.io as sio
import cv2
import numpy as np
import matplotlib.pyplot as plt


M_CIE_RGB2XYZ = np.array([
    [ .490, .310, .20],
    [.177,  .813,  .011],
    [ .000, -0.010,  .990]
])

M_CIE_XYZ2RGB = np.linalg.inv(M_CIE_RGB2XYZ)

def convert_cie_xzy_to_srgb(c):
    # convert to cie rgb
    cie_rgb = c @ M_CIE_XYZ2RGB
    cie_rgb = np.clip(cie_rgb, 0, None)
    # gamma correction
    gamma = 2.4
    srgb = np.array([ ((12.92 * x) if (x < 0.0031308) else ( 1.055*(x**(1/gamma)) - 0.055)) for x in cie_rgb])
    srgb = np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
    return srgb


data = sio.loadmat("visibleSpectrum.mat")
print(type(data))
print(data)

CMFs = data['CMFs']
wavelengths = CMFs[:, 0]
cie_xyz = CMFs[:, 1:]
colors = np.array([[convert_cie_xzy_to_srgb(x) for x in cie_xyz]])
plt.figure("Spectrum", (10, 5))
plt.imshow(colors, extent=(380, 760, 0, 50), aspect="auto")
plt.xlabel("Wavelength (nm)")
plt.xticks([380, 760])
plt.yticks([])
plt.show()