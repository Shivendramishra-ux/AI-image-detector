import cv2
import numpy as np
from PIL import Image
import os

# ---------- LOAD ----------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return img, gray

# ---------- PRNU ----------
def prnu_metrics(gray):
    smooth = cv2.GaussianBlur(gray, (7,7), 0)
    noise = gray - smooth

    sigma = np.std(noise)
    corr = np.corrcoef(noise[:-1].flatten(), noise[1:].flatten())[0,1]

    return sigma, corr

# ---------- FFT ----------
def fft_metrics(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)

    h, w = mag.shape
    cx, cy = h//2, w//2
    mag[cx-5:cx+5, cy-5:cy+5] = 0

    mean = np.mean(mag)
    std = np.std(mag)

    threshold = mean + 2*std
    peaks = mag > threshold

    peak_ratio = np.sum(peaks) / mag.size

    return mean, std, peak_ratio

# ---------- TEXTURE ----------
def texture_metrics(gray):
    hist = cv2.calcHist([gray.astype('float32')],[0],None,[256],[0,1])
    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist + 1e-9))
    return entropy

# ---------- CONSISTENCY ----------
def consistency_metrics(gray):
    h, w = gray.shape
    size = 64
    patches = []

    for y in range(0, h-size, size):
        for x in range(0, w-size, size):
            patch = gray[y:y+size, x:x+size]
            patches.append(np.std(patch))

    return np.std(patches)

# ---------- SMOOTHNESS ----------
def smoothness_metrics(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

# ---------- CHROMA ----------
def chroma_metrics(img):
    b, g, r = cv2.split(img)
    corr = np.corrcoef(r.flatten(), g.flatten())[0,1]
    return corr

# ---------- EXIF ----------
def exif_metrics(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        count = len(exif) if exif else 0
    except:
        count = 0
    return count

def ela_metrics(path):
    try:
        img = Image.open(path).convert("RGB")

        temp_path = "temp_ela.jpg"
        img.save(temp_path, "JPEG", quality=75)

        comp = Image.open(temp_path)

        diff = np.abs(np.array(img) - np.array(comp))
        std = np.std(diff)

        return std

    except Exception as e:
        print("ELA error:", e)
        return 0

# ---------- TEST ----------
def test_folder(folder_path):
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if not file.lower().endswith(('.jpg','.png','.jpeg','.webp')):
            continue

        img, gray = load_image(path)

        print("\n==========================")
        print("IMAGE:", file)

        sigma, corr = prnu_metrics(gray)
        print("PRNU sigma:", round(sigma,5), "corr:", round(corr,3))

        mean, std, peak = fft_metrics(gray)
        print("FFT mean:", round(mean,3), "std:", round(std,3), "peak_ratio:", round(peak,5))

        entropy = texture_metrics(gray)
        print("Texture entropy:", round(entropy,3))

        consistency = consistency_metrics(gray)
        print("Consistency:", round(consistency,5))

        smooth = smoothness_metrics(gray)
        print("Smoothness (lap var):", round(smooth,2))

        chroma = chroma_metrics(img)
        print("Chroma corr:", round(chroma,4))

        exif = exif_metrics(path)
        print("EXIF fields:", exif)

        ela = ela_metrics(path)
        print("ELA std:", round(ela, 3))
        

folder = "test_images"   
test_folder(folder)