# Generation of a Noisy Point Set Conforming to a Line and a Circle
import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import matplotlib.pyplot as plt
#import tikzplotlib

# np.random.seed(0)
N = 100
half_n = N // 2
r = 10
x0_gt, y0_gt = 2, 3  # Center

s = r / 16
t = np.random.uniform(0, 2 * np.pi, half_n)
n = s * np.random.randn(half_n)
x, y = x0_gt + (r + n) * np.cos(t), y0_gt + (r + n) * np.sin(t)
X_circ = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

s = 1.
m, b = -1, 2
x = np.linspace(-12, 12, half_n)
y = m * x + b + s * np.random.randn(half_n)
X_line = np.hstack((x.reshape(half_n, 1), y.reshape(half_n, 1)))

X = np.vstack((X_circ, X_line))

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X_line[:, 0], X_line[:, 1], label='Line')
ax.scatter(X_circ[:, 0], X_circ[:, 1], label='Circle')

circle_gt = plt.Circle(
    (x0_gt, y0_gt), r,
    color='g', fill=False,
    label='Ground truth circle'
)
ax.add_patch(circle_gt)
ax.plot(x0_gt, y0_gt, '+', color='g')

x_min, x_max = ax.get_xlim()
x_ = np.array([x_min, x_max])
y_ = m * x_ + b
plt.plot(x_, y_, color='m', label='Ground truth line')
plt.legend()
plt.show()

#-------------RANSAC-------------#

p = 0.99  # probability of success
e = 0.5  # expected outlier ratio
s_line = 2  # number of points needed to fit a line
s_circ = 3  # number of points needed to fit a circle
N_line = int(np.log(1 - p) / np.log(1 - (1 - e)**s_line)) #Number of iterations for line
N_circ = int(np.log(1 - p) / np.log(1 - (1 - e)**s_circ)) #Number of iterations for circle
T_line = 1.5 # threshold for line(std is 1)
T_circ = 1.5 * 10/16 # threshold for circle(std is 10/16)

def calculate_shortes_diatance(x_i, y_i, a_i, b_i, d_i):
    return abs((a_i * x_i) + (b_i * y_i) - d_i)

def RANSAC_for_line(N, points, T):
    best_fit = [0, 0, 0]
    highest_no_inliers = 0
    best_inlier_indices = [] #for Part 2(b)
    
    
    for i in range(N): 
        indices = np.random.choice(len(points), size=2, replace=False)

        p1_r = points[indices[0]]
        p2_r = points[indices[1]]
        x1_r, y1_r = p1_r[0], p1_r[1]
        x2_r, y2_r = p2_r[0], p2_r[1]
        
        dx = x2_r - x1_r
        dy = y2_r - y1_r
        norm = np.sqrt((dx * dx) + (dy * dy))
        
        if norm == 0: # Prevent dividing by zero if points are identical
            continue 
            
        a = -dy / norm
        b = dx / norm
        d = (a * x1_r) + (b * y1_r)
        
        no_inliers = 0
        current_inlier_indices = [] #track indices for this iteration
        
        for j in range(len(points)):
            x = points[j][0]
            y = points[j][1]
            distance = calculate_shortes_diatance(x, y, a, b, d)
            
            if distance < T:
                no_inliers += 1
                current_inlier_indices.append(j) # Remember this point is an inlier
                
        if highest_no_inliers < no_inliers:
            highest_no_inliers = no_inliers
            best_fit = [a, b, d] # FIXED: use d, not c
            best_inlier_indices = current_inlier_indices # Save the winning points
            best_sample = [p1_r, p2_r]

    # --- Calculate best fitting line considering all the inliers ---
    if len(best_inlier_indices) > 2:
        # 1. Get the actual (x, y) coordinates of all the winning inliers
        inliers = points[best_inlier_indices]
        
        # 2. Find their center of mass (mean x and mean y)
        centroid = np.mean(inliers, axis=0)
        
        # 3. Shift the points so their center is at (0,0)
        centered = inliers - centroid
        
        # 4. Use SVD to find the principal directions
        U, S, Vt = np.linalg.svd(centered)
        
        # 5. The smallest direction (last row of Vt) is our perfectly refined normal [a, b]
        a_ref = Vt[-1, 0]
        b_ref = Vt[-1, 1]
        
        # 6. Calculate the new exact 'd' using the centroid
        d_ref = (a_ref * centroid[0]) + (b_ref * centroid[1])
        
        # Overwrite our old variables with the perfect ones!
        best_fit = [a_ref, b_ref, d_ref]
    
    print("Best fitting line is " + str(round(best_fit[0], 2)) + "x + " + str(round(best_fit[1], 2)) + "y = " + str(round(best_fit[2], 2)))
          
    return best_fit, best_inlier_indices, np.array(best_sample)

def RANSAC_for_circle(N, points, T): #Assume circle equation is x^2 + y^2 + ax + by + c = 0, where a, b, c are the parameters to be estimated
    best_fit = [0, 0, 0, 0] # a,b, c, d
    highest_no_inliers = 0
    best_inlier_indices = [] 
    
    for i in range(N): 
        indices = np.random.choice(len(points), size=3, replace=False)

        p1_r = points[indices[0]]
        p2_r = points[indices[1]]
        p3_r = points[indices[2]]
        x1_r, y1_r = p1_r[0], p1_r[1]
        x2_r, y2_r = p2_r[0], p2_r[1]
        x3_r, y3_r = p3_r[0], p3_r[1]
        
        D = 2 * (x1_r * (y2_r - y3_r) + x2_r * (y3_r - y1_r) + x3_r * (y1_r - y2_r))
        # Calculate the squared magnitudes of the points from the origin
        s1 = x1_r**2 + y1_r**2
        s2 = x2_r**2 + y2_r**2
        s3 = x3_r**2 + y3_r**2

        # Calculate center coordinates (h, k)
        h = (s1 * (y2_r - y3_r) + s2 * (y3_r - y1_r) + s3 * (y1_r - y2_r)) / D
        k = (s1 * (x3_r - x2_r) + s2 * (x1_r - x3_r) + s3 * (x2_r - x1_r)) / D

        # Calculate radius using the distance from the center (h, k) to any point (e.g., p1)
        radius = np.sqrt((x1_r - h)**2 + (y1_r - k)**2)

        
        no_inliers = 0
        current_inlier_indices = [] #track indices for this iteration
        
        for j in range(len(points)):
            x = points[j][0]
            y = points[j][1]
            radial_error = np.sqrt((x - h)**2 + (y - k)**2) - radius
            distance = abs(radial_error)
            
            if distance < T:
                no_inliers += 1
                current_inlier_indices.append(j) # Remember this point is an inlier
                
        if highest_no_inliers < no_inliers:
            highest_no_inliers = no_inliers
            best_fit = [h, k, radius] 
            best_inlier_indices = current_inlier_indices # Save the winning points
            best_sample = [p1_r, p2_r, p3_r]
    
    print("Best fitting circle is centered at (" + str(round(best_fit[0], 2)) + ", " + str(round(best_fit[1], 2)) + ") with radius " + str(round(best_fit[2], 2)))
          
    return best_fit, best_inlier_indices, np.array(best_sample)



#-------------------plotting-------------------#
# 1. Run Line RANSAC on the full dataset X
best_line, line_inliers_idx, line_sample = RANSAC_for_line(N_line, X, T_line)
line_inliers = X[line_inliers_idx] # Get the actual points

# 2. Subtract line inliers to get the remnant for the circle
X_remnant = np.delete(X, line_inliers_idx, axis=0)

# 3. Run Circle RANSAC on the remnant
best_circ, circ_inliers_idx, circ_sample = RANSAC_for_circle(N_circ, X_remnant, T_circ)
circ_inliers = X_remnant[circ_inliers_idx] # Get the actual points

# ==========================================
# PLOTTING (Part 2c)
# ==========================================
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# 1. Plot ALL points as a base layer (blue dots)
ax.scatter(X[:, 0], X[:, 1], color='tab:blue', s=25, label='All points')

# 2. Plot Inliers over the top
ax.scatter(line_inliers[:, 0], line_inliers[:, 1], color='yellowgreen', s=25, label='Line inliers')
ax.scatter(circ_inliers[:, 0], circ_inliers[:, 1], color='c', s=25, label='Circle inliers')

# 3. Plot the specific Best Samples used to generate the models
ax.scatter(line_sample[:, 0], line_sample[:, 1], color='m', marker='x', s=50, label='Best sample for line')
ax.scatter(circ_sample[:, 0], circ_sample[:, 1], color='r', marker='o', s=30, label='Best sample for circle')

# 4. Draw Ground Truths (from Listing 1)
circle_gt = plt.Circle((x0_gt, y0_gt), r, color='g', fill=False, linewidth=1, label='Ground truth circle')
ax.add_patch(circle_gt)
ax.plot(x0_gt, y0_gt, '+', color='g') # Center cross
x_min, x_max = ax.get_xlim()
x_arr = np.array([x_min, x_max])
y_gt = m * x_arr + b
ax.plot(x_arr, y_gt, color='tab:blue', linewidth=1, label='Ground truth line')

# 5. Draw RANSAC Line (ax + by = d -> y = (d - ax) / b)
a_L, b_L, d_L = best_line
y_ransac_line = (d_L - a_L * x_arr) / b_L
ax.plot(x_arr, y_ransac_line, color='m', linewidth=1, label='RANSAC line')

# 6. Draw RANSAC Circle
h_C, k_C, r_C = best_circ
ransac_circle_patch = plt.Circle((h_C, k_C), r_C, color='m', fill=False, linewidth=1, label='RANSAC circle')
ax.add_patch(ransac_circle_patch)

# Final formatting to match Figure 1
ax.set_aspect('equal') # Ensures circles look round, not oval
plt.legend(loc='upper right', framealpha=1.0)
plt.show()