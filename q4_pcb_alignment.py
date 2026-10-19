
import cv2 as cv
import numpy as np
import os
import glob

base = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(base, "images", "pcbs")
out = os.path.join(base, "q4_results")
os.makedirs(out, exist_ok=True)

reference = cv.imread(os.path.join(folder, "AECC1.jpg"))


gray1 = cv.cvtColor(reference, cv.COLOR_BGR2GRAY)
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
matcher = cv.BFMatcher()

h, w = gray1.shape

for path in glob.glob(os.path.join(folder, "*.jpg")):
    name = os.path.basename(path)

    if name.lower() == "aecc1.jpg":
        continue

    image = cv.imread(path)

    if image is None:
        continue

    gray2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print(name, "- descriptors unavailable")
        continue

    pairs = matcher.knnMatch(des1, des2, k=2)

    good = [
        m for pair in pairs if len(pair) == 2
        for m, n in [pair]
        if m.distance < 0.65 * n.distance
    ]

    if len(good) < 8:
        print(name, "- insufficient matches")
        continue

    p1 = np.float32(
        [kp1[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    p2 = np.float32(
        [kp2[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, status = cv.findHomography(
        p2, p1, cv.RANSAC, 4.0
    )

    if H is None or status is None or np.sum(status) < 8:
        print(name, "- unreliable homography")
        continue

    corners = np.float32([
        [0, 0],
        [image.shape[1] - 1, 0],
        [image.shape[1] - 1, image.shape[0] - 1],
        [0, image.shape[0] - 1]
    ]).reshape(-1, 1, 2)

    warped_corners = cv.perspectiveTransform(
        corners, H
    ).reshape(-1, 2)

    if not np.isfinite(warped_corners).all():
        print(name, "- invalid transformation")
        continue

    if (
        np.max(np.abs(warped_corners)) > 5 * max(h, w)
        or abs(np.linalg.det(H)) < 1e-10
    ):
        print(name, "- implausible transformation")
        continue

    inliers = [
        m for m, ok in zip(good, status.ravel()) if ok
    ]

    aligned = cv.warpPerspective(image, H, (w, h))

    valid = cv.warpPerspective(
        np.ones(image.shape[:2], dtype=np.uint8),
        H, (w, h),
        flags=cv.INTER_NEAREST
    )

    overlap = cv.addWeighted(
        reference, 0.5, aligned, 0.5, 0
    )
    overlap[valid == 0] = reference[valid == 0]

    # Difference image with improved visibility
    ref_blur = cv.GaussianBlur(reference, (5, 5), 0)
    aligned_blur = cv.GaussianBlur(aligned, (5, 5), 0)

    d = cv.absdiff(ref_blur, aligned_blur)
    d = cv.cvtColor(d, cv.COLOR_BGR2GRAY)

    valid_pixels = valid > 0
    diff = np.zeros_like(d)

    if np.any(valid_pixels):
        hi = np.percentile(d[valid_pixels], 99.5)

        if hi > 0:
            diff[valid_pixels] = np.clip(
                d[valid_pixels].astype(np.float32) * 255 / hi,
                0, 255
            ).astype(np.uint8)

    diff_color = cv.applyColorMap(diff, cv.COLORMAP_JET)
    diff_color[~valid_pixels] = 0

    # Display only 20 strong RANSAC inlier matches
    shown_matches = sorted(
        inliers, key=lambda m: m.distance
    )[:20]

    matches_img = cv.drawMatches(
        reference, kp1,
        image, kp2,
        shown_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save results
    stem = os.path.splitext(name)[0]

    cv.imwrite(os.path.join(out, stem + "_matches.jpg"), matches_img)
    cv.imwrite(os.path.join(out, stem + "_aligned.jpg"), aligned)
    cv.imwrite(os.path.join(out, stem + "_overlap.jpg"), overlap)
    cv.imwrite(os.path.join(out, stem + "_difference.jpg"), diff_color)
    cv.imwrite(os.path.join(out, stem + "_difference_gray.jpg"), diff)

    print(
        name,
        "- matches:", len(good),
        "- RANSAC inliers:", len(inliers)
    )

print("Finished. Results:", out)
