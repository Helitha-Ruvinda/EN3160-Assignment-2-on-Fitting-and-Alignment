import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the two images and choose the four corners of the target surface in the scene image
scene = cv2.imread("images/EmptyBook.png")
flag = cv2.imread("images/Harry_Potter_and_the_Order_of_the_Phoenix.jpg")
plt.imshow(cv2.cvtColor(scene, cv2.COLOR_BGR2RGB))
plt.title("Select four corners of the target surface")
plt.show()

h, w = flag.shape[:2]

# Four corners of the flag
src = np.float32([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
])

dst = np.float32([
    [110, 171],  # Top-left
    [507, 69],  # Top-right
    [730, 542],  # Bottom-right
    [275, 675]   # Bottom-left
])

# Calculate the homography
H = cv2.getPerspectiveTransform(src, dst)

# Warp the flag into the scene
warped = cv2.warpPerspective(
    flag, H, (scene.shape[1], scene.shape[0])
)

# Create a mask so the flag's black background
mask = np.zeros((h, w), dtype=np.uint8)
cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)

# Simpler overlay using the transformed flag's polygon
warped_mask = cv2.warpPerspective(
    np.ones((h, w), dtype=np.uint8) * 255,
    H, (scene.shape[1], scene.shape[0])
)

result = scene.copy()
region = warped_mask > 0
result[region] = warped[region]

plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

cv2.imwrite("images/Harry_Potter.png", result)