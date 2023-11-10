import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2hed, hed2rgb
import cv2

def color_deconv(img):
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Separate the stains from the IHC image
    ihc_hed = rgb2hed(rgb_image)

    # Create an RGB image for each of the stains
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))

    # Display
    fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(rgb_image)
    ax[0].set_title("Original image")

    ax[1].imshow(ihc_h)
    ax[1].set_title("Hematoxylin")

    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosin")  # Note that there is no Eosin stain in this image

    ax[3].imshow(ihc_d)
    ax[3].set_title("DAB")

    for a in ax.ravel():
        a.axis('off')

    fig.tight_layout()

    plt.savefig('color_deconv.png')
    #plt.show()
    return ihc_d,ihc_h,ihc_hed #DAB,HE,hed


