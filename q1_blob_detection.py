import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, binary_fill_holes, label

im = cv.imread('images/the_berry_farms_sunflower_field.jpeg', cv.IMREAD_REDUCED_COLOR_4)

# 1. Yellow map and binary mask (petals = 1, everything else = 0)
b, g, r = cv.split(im.astype(np.float32) / 255)
yellow = np.minimum(r, g) - b
yellow_bin = yellow > 0.25
mask = yellow_bin.astype(np.uint8)
mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # seal gaps between petals

# 2. Fill the dark centres (small holes only)
filled = binary_fill_holes(mask)
holes, n = label(filled & (mask == 0))
areas = np.bincount(holes.ravel())
small = areas < 900
small[0] = False
mask = (mask.astype(bool) | small[holes]).astype(np.float32)

# 3. Scale-normalized LoG (bright blobs) over a range of sigmas
sigmas = np.arange(4, 36, 1.5)
stack = []
for s in sigmas:
    blur = cv.GaussianBlur(mask, (0, 0), s)
    log = cv.Laplacian(blur, cv.CV_32F)
    stack.append(-(s**2) * log)
stack = np.array(stack)

# 4. Scale-space extrema
local_max = maximum_filter(stack, size=(3, 3, 3))
threshold = 0.3
k, y, x = np.nonzero((stack == local_max) & (stack > threshold))

# 5. Strongest first, drop circles whose centre lies inside an already kept circle
order = np.argsort(-stack[k, y, x])
circles = []
for i in order:
    cx, cy, rad = x[i], y[i], np.sqrt(2) * sigmas[k[i]]
    ok = True
    for (px, py, pr) in circles:
        if np.hypot(cx - px, cy - py) < max(rad, pr):
            ok = False
            break
    if ok:
        circles.append((cx, cy, rad))

# 6. Validate: the ring of a real flower is mostly yellow petals
YY, XX = np.ogrid[:im.shape[0], :im.shape[1]]
final = []
for (cx, cy, rad) in circles:
    dist = np.sqrt((XX - cx)**2 + (YY - cy)**2)
    ring = (dist > 0.5 * rad) & (dist < 1.0 * rad)
    if ring.sum() > 0 and yellow_bin[ring].mean() > 0.5:
        final.append((cx, cy, rad))
circles = final

# 7. Draw
out = im.copy()
for (cx, cy, rad) in circles:
    cv.circle(out, (int(cx), int(cy)), int(rad), (0, 0, 255), 2)

# 8. Report
circles.sort(key=lambda c: c[2], reverse=True)
print("Sigma range:", sigmas[0], "to", sigmas[-1])
print("Number of circles:", len(circles))
print("Largest circles (x, y, radius):")
for c in circles[:5]:
    print(c)

plt.figure(figsize=(10, 10))
plt.imshow(cv.cvtColor(out, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.show()