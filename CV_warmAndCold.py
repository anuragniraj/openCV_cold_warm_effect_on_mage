import numpy as np
from scipy.interpolate import UnivariateSpline
import cv2 as cv, glob, os, shutil

img_path = "Paste_the_path_here"
save_path = "Paste_the_path_here"

os.makedirs(save_path, exist_ok=True)

class ColorEffect():
    def __init__(self):
        """Initialize look-up tables for curve filter"""
        # create look-up tables for increasing and decreasing a channel
        # Warm effect
        self.incr_red_warm_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                        [0, 140, 200, 230, 256])  # Increased values for more warmth
        self.decr_blue_warm_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                        [0, 30, 80, 120, 192])
        # Cold effect
        self.decr_red_cold_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                        [0, 70, 140, 210, 256])
        self.incr_blue_cold_lut = self._create_LUT_8UC1([0, 64, 128, 192, 256],
                                                        [0, 140, 200, 230, 256])  # Increased values for more cold

    def apply_warm_effect(self, img_rgb):
        c_r, c_g, c_b = cv.split(img_rgb)
        c_r = cv.LUT(c_r, self.incr_red_warm_lut).astype(np.uint8)
        c_b = cv.LUT(c_b, self.decr_blue_warm_lut).astype(np.uint8)
        img_rgb = cv.merge((c_r, c_g, c_b))

        # increase color saturation
        c_h, c_s, c_v = cv.split(cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV))
        c_s = cv.LUT(c_s, self.decr_blue_warm_lut).astype(np.uint8)
        return cv.cvtColor(cv.merge((c_h, c_s, c_v)), cv.COLOR_HSV2RGB)

    def apply_cold_effect(self, img_rgb):
        c_r, c_g, c_b = cv.split(img_rgb)
        c_r = cv.LUT(c_r, self.decr_red_cold_lut).astype(np.uint8)
        c_b = cv.LUT(c_b, self.incr_blue_cold_lut).astype(np.uint8)
        img_rgb = cv.merge((c_r, c_g, c_b))

        # decrease color saturation
        c_h, c_s, c_v = cv.split(cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV))
        c_s = cv.LUT(c_s, self.incr_blue_cold_lut).astype(np.uint8)
        return cv.cvtColor(cv.merge((c_h, c_s, c_v)), cv.COLOR_HSV2RGB)

    def _create_LUT_8UC1(self, x, y):
        """Creates a look-up table using scipy's spline interpolation"""
        spl = UnivariateSpline(x, y)
        return spl(range(256))


for img_file in glob.glob(os.path.join(img_path, '*.png')):
    
    filename = os.path.basename(img_file)
    img = cv.imread(img_file)
    xmlFile = img_file[:-4] + '.json'
    seeImg = cv.resize(img, (500, 500))

    # Apply effects
    effect = ColorEffect()
    Warm = effect.apply_warm_effect(img)
    cv.imwrite(os.path.join(save_path, f"{filename[:-4]}_warm{filename[-4:]}"), Warm)
    shutil.copy(xmlFile, os.path.join(save_path, f"{filename[:-4]}_warm{xmlFile[-5:]}"))
    
    
    Cold = effect.apply_cold_effect(img)
    cv.imwrite(os.path.join(save_path, f"{filename[:-4]}_cold{filename[-4:]}"), Cold)
    shutil.copy(xmlFile, os.path.join(save_path, f"{filename[:-4]}_cold{xmlFile[-5:]}"))
    # Display the original and processed images
    
    seeWarm = cv.resize(Warm, (500, 500))
    seeCold = cv.resize(Cold, (500, 500))
    show = cv.hconcat([seeImg, seeWarm, seeCold])
    cv.imshow("image", show)
    cv.waitKey(1)
cv.destroyAllWindows()
