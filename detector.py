import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

# ---------- LOAD ----------
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    return img, gray

# ---------- PRNU ----------
def prnu_score(gray):
    smooth = cv2.GaussianBlur(gray, (7,7), 0)
    noise = gray - smooth
    sigma = np.std(noise)

    # autocorrelation
    corr = np.corrcoef(noise[:-1].flatten(), noise[1:].flatten())[0,1]

    score = 0.7

    if sigma < 0.04:
        score += 0.38
    elif sigma < 0.045:
        score += 0.18

    if abs(corr) > 0.15:
        score += 0.22

    return min(score, 1)

# ---------- FFT ----------
def fft_score(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1)

    # remove center
    h, w = mag.shape
    cx, cy = h // 2, w // 2
    mag[cx-5:cx+5, cy-5:cy+5] = 0

    mean = np.mean(mag)
    std = np.std(mag)

    threshold = mean + 2 * std
    peaks = mag > threshold
    peak_ratio = np.sum(peaks) / mag.size

    score = 0.5

    # 🔥 PEAK (strongest)
    if peak_ratio > 0.036:
        score += 0.5
    elif peak_ratio > 0.028:
        score += 0.3

    # 🔥 STD (medium)
    if std < 0.72 or std > 1.6:
        score += 0.25
    elif std < 0.79 or std > 1.4:
        score += 0.15

    # 🔥 MEAN (weak)
    if mean < 0.5 or (2.25 < mean < 2.50):
        score += 0.15

    return float(min(score, 1))

# ---------- TEXTURE ----------
def texture_score(gray):
    # histogram entropy (correct way)
    hist = cv2.calcHist([gray.astype('float32')],[0],None,[256],[0,1])
    hist = hist / np.sum(hist)

    entropy = -np.sum(hist * np.log2(hist + 1e-9))

    score = 0.0

    # 🔥 LOW entropy → over-smooth AI
    if entropy < 5.5:
        score += 0.4
    elif entropy < 6.3:
        score += 0.25

    # 🔥 VERY HIGH entropy → synthetic noise (rare but possible)
    elif entropy > 7.8:
        score += 0.3

    # 🔥 Normal range → real-like
    # (5.8 – 7.5 is typical real zone)

    return float(min(score, 1))

# ---------- EXIF ----------

from PIL.ExifTags import TAGS

def exif_details_score(path):
    try:
        img = Image.open(path)
        exif_data = img.getexif()

        useful_tags = [
            "Make", "Model", "DateTime",
            "Software", "LensModel",
            "ISOSpeedRatings", "ExposureTime",
            "FNumber" , "GPS"
        ]

        details = {}

        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)

            # only keep useful + readable values
            if tag in useful_tags and isinstance(value, (str, int, float)):
                details[tag] = value

        # -------- scoring --------
        score = 2

        if "Make" in details and "Model" in details:
            score -= 0.5

        if "DateTime" in details:
            score -= 0.3
        
        if "Software" in details:
            score -= 0.7

        if len(details) == 0:
            score += 0.2

        if "LensModel" in details or "ExposureTime" in details or "FNumber" in details :
            score -= 0.3

        return float(min(max(score, 0), 1)), details

    except Exception as e:
        print("EXIF error:", e)
        return 0.5, {}

# ---------- ELA ----------
def ela_score(path):
    try:
        img = Image.open(path).convert("RGB")

        temp_path = "temp_ela.jpg"
        img.save(temp_path, "JPEG", quality=75)

        comp = Image.open(temp_path)

        # difference
        diff = np.abs(np.array(img) - np.array(comp))

        # convert to grayscale-like intensity
        diff_gray = np.mean(diff, axis=2)

        mean = np.mean(diff_gray)
        std = np.std(diff_gray)

        # ---- scoring ----
        score = 0.0

        # low variation → suspicious (AI)
        if std < 5:
            score += 0.5
        elif std < 8:
            score += 0.3

        # extremely low mean → too clean
        if mean < 2:
            score += 0.3

        return float(min(score, 1))

    except Exception as e:
        print("ELA error:", e)
        return 0.0

# ---------- CONSISTENCY ----------
def consistency_score(gray):
    h, w = gray.shape
    size = 64
    patches = []

    for y in range(0, h-size, size):
        for x in range(0, w-size, size):
            patch = gray[y:y+size, x:x+size]
            patches.append(np.std(patch))

    consistency = np.std(patches)

    score = 1 - consistency * 10

    return min(max(score, 0), 1)

# ---------- SMOOTHNESS ----------
def smoothness_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var = lap.var()

    score = 1 / (var + 10)
    return float(min(max(score, 0), 1))
   
# ---------- CHROMATIC ----------
def chroma_score(img):
    b, g, r = cv2.split(img)

    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0,1]

    score = (corr_rg - 0.9) * 5

    return min(max(score, 0), 1)
# ---------- DETECT ----------
def detect(path):
    img, gray = load_image(path)

    exif_score_val, exif_info = exif_details_score(path)

    scores = {
    "PRNU": prnu_score(gray),
    "FFT": fft_score(gray),
    "Texture": texture_score(gray),
    "EXIF": exif_score_val,
    "ELA": ela_score(path),
    "Consistency": consistency_score(gray),
    "Smoothness": smoothness_score(gray),
    "Chroma": chroma_score(img)
}
   
    weights = {
    "PRNU": 0.20,
    "FFT": 0.10,
    "Texture": 0.05,
    "EXIF": 0.15,
    "ELA": 0.10,
    "Consistency": 0.14,
    "Smoothness": 0.01,
    "Chroma": 0.15
}

    final = sum(scores[k] * weights[k] for k in scores)
    
    # ---------- MULTI-SIGNAL VOTING ----------
    ai_votes = sum(1 for s in scores.values() if s > 0.6)
  # -------- SIGNAL COUNTS --------
    strong_features = ["PRNU", "FFT", "Consistency"]
    medium_features = ["Texture", "Smoothness", "Chroma"]
    weak_features = ["EXIF", "ELA"]

    strong_signals = sum(1 for k in strong_features if scores[k] > 0.8)
    medium_signals = sum(1 for k in medium_features if scores[k] > 0.6)
    weak_signals = sum(1 for k in weak_features if scores[k] > 0.6)

    #  -------- SCORE ADJUSTMENT --------
    if medium_signals >= 2:
      final += 0.12
    elif medium_signals == 1:
        final += 0.05

    if weak_signals >= 2:
        final += 0.05   # reduced impact

# clamp final
    final = min(final, 1)

# -------- FINAL DECISION --------
    if strong_signals >= 2:
        result = "AI GENERATED"

    elif strong_signals >= 1 and medium_signals >= 2:
        result = "AI GENERATED"

    elif final > 0.65:
        result = "AI GENERATED"

    elif final > 0.55:
        result = "SUSPICIOUS"

    else:
        result = "REAL"

    return scores, final, result , exif_info