import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv

def read_csv(csv_path):
    """Reads a CSV file and organizes the points into paths."""
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def detect_shape(XY):
    """Detects the shape type from the provided points."""
    contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    area = cv2.contourArea(contour)
    
    if area < 100:
        return " "

    if is_straight_line(XY):
        return "Straight Line"

    if len(approx) == 4:
        if is_square(contour):
            return "Square"
        else:
            return "Rectangle"

    elif len(approx) > 5:
        if is_star(approx):
            return "Star"
        
        circularity = 4 * np.pi * area / (peri * peri)
        
        if circularity > 0.95:
            return "Circle"
        elif 0.8 <= circularity < 0.95:
            ellipse = cv2.fitEllipse(contour)
            major_axis, minor_axis = max(ellipse[1]), min(ellipse[1])
            if major_axis / minor_axis > 1.05:
                return "Ellipse"
            else:
                return "Circle"
    
    return " "

def is_square(contour, tolerance=0.05):
    """Checks if the contour forms a square."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    aspect_ratio = min(width, height) / max(width, height)

    contour_area = cv2.contourArea(contour)
    rect_area = width * height
    area_ratio = contour_area / rect_area

    is_square_shape = abs(1 - aspect_ratio) < tolerance
    is_filled = area_ratio > (1 - tolerance)

    return is_square_shape and is_filled

def is_star(approx):
    """Checks if the contour forms a star shape."""
    num_vertices = len(approx)
    if num_vertices > 8:
        angles = []
        for i in range(num_vertices):
            p1 = approx[i][0]
            p2 = approx[(i + 1) % num_vertices][0]
            p3 = approx[(i + 2) % num_vertices][0]
            angle = calculate_angle(p1, p2, p3)
            angles.append(angle)
        
        sharp_angles = [angle for angle in angles if angle < 60 or angle > 300]
        if len(sharp_angles) >= 5:
            return True

    return False

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    cb = b - c
    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def is_straight_line(XY, tolerance=0.05):
    """Checks if the points form a straight line."""
    if len(XY) < 3:
        return True
    
    total_length = np.sum(np.sqrt(np.sum(np.diff(XY, axis=0)**2, axis=1)))
    start_to_end = np.linalg.norm(XY[-1] - XY[0])
    return (total_length - start_to_end) / total_length < tolerance

def regularize_circle(XY):
    """Regularizes the points to form a perfect circle."""
    xc, yc, r = kasa_circle_fitting(XY)
    t = np.linspace(0, 2 * np.pi, 100)
    x_fit = xc + r * np.cos(t)
    y_fit = yc + r * np.sin(t)
    return np.vstack((x_fit, y_fit)).T

def kasa_circle_fitting(XY):
    """Fits a circle to the points using Kasa's method."""
    x = XY[:, 0]
    y = XY[:, 1]
    A = np.array([x, y, np.ones_like(x)]).T
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def regularize_ellipse(XY):
    """Regularizes the points to form a perfect ellipse."""
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    width = x_max - x_min
    height = y_max - y_min
    t = np.linspace(0, 2 * np.pi, 200)
    x = center[0] + width/2 * np.cos(t)
    y = center[1] + height/2 * np.sin(t)
    
    return np.column_stack((x, y))

def regularize_rectangle(XY):
    """Regularizes the points to form a perfect rectangle."""
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)

    regularized = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
        [x_min, y_min]  
    ])
    
    return regularized

def regularize_square(XY):
    """Regularizes the points to form a perfect square."""
    x_min, y_min = np.min(XY, axis=0)
    x_max, y_max = np.max(XY, axis=0)

    side_length = min(x_max - x_min, y_max - y_min)

    reg_XY = np.array([
        [x_min, y_min],
        [x_min + side_length, y_min],
        [x_min + side_length, y_min + side_length],
        [x_min, y_min + side_length],
        [x_min, y_min]  
    ])
    
    return reg_XY

def regularize_star(XY, num_points=5):
    """Regularizes the points to form a perfect star."""
    centroid = np.mean(XY, axis=0)

    radii = np.linalg.norm(XY - centroid, axis=1)
    outer_radius = np.max(radii)
    inner_radius = np.min(radii)
    
    angles = np.linspace(0, 2*np.pi, 2*num_points, endpoint=False)
    
    star_points = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = centroid[0] + r * np.cos(angle)
        y = centroid[1] + r * np.sin(angle)
        star_points.append([x, y])
    
    star_points.append(star_points[0])  
    
    return np.array(star_points)

def regularize_straight_line(XY):
    """Regularizes the points to form a perfect straight line."""
    return np.array([XY[0], XY[-1]])

def detect_and_regularize_shape(XY):
    """Detects the shape and returns the regularized points."""
    if is_straight_line(XY):
        return "Straight Line", regularize_straight_line(XY)
    
    shape = detect_shape(XY)
    if shape == "Circle":
        regularized_XY = regularize_circle(XY)
    elif shape == "Ellipse":
        regularized_XY = regularize_ellipse(XY)
    elif shape == "Rectangle":
        regularized_XY = regularize_rectangle(XY)
    elif shape == "Square":
        regularized_XY = regularize_square(XY)
    elif shape == "Star":
        regularized_XY = regularize_star(XY)
    else:
        regularized_XY = XY
    return shape, regularized_XY

def calculate_symmetry_lines(XY, shape_type):
    """Calculates symmetry lines for the given shape."""
    if shape_type in ["Circle", "Ellipse", "Square", "Rectangle"]:
        centroid = np.mean(XY, axis=0)
        if shape_type == "Circle" or shape_type == "Square":
            return [
                ([centroid[0], np.min(XY[:, 1])], [centroid[0], np.max(XY[:, 1])]),  # Vertical
                ([np.min(XY[:, 0]), centroid[1]], [np.max(XY[:, 0]), centroid[1]])   # Horizontal
            ]
        elif shape_type == "Ellipse" or shape_type == "Rectangle":
            return [
                ([np.min(XY[:, 0]), centroid[1]], [np.max(XY[:, 0]), centroid[1]])   # Horizontal
            ]
    elif shape_type == "Star":
        centroid = np.mean(XY, axis=0)
        angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
        lines = []
        for angle in angles:
            x = centroid[0] + np.cos(angle) * 1000
            y = centroid[1] + np.sin(angle) * 1000
            lines.append(([centroid[0], centroid[1]], [x, y]))
        return lines
    return []

def plot_shapes(path_XYs, regularized_XYs):
    """Plots original and regularized shapes with symmetry lines, and prints information about detected shapes."""
    fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    for a in ax:
        a.set_aspect('equal')

    # Print shapes detected and regularized in bold
    print("\033[1mShapes Detected and Regularized:\033[0m")

    for i, (orig_XYs, reg_XYs) in enumerate(zip(path_XYs, regularized_XYs)):
        color = colors[i % len(colors)]
        for XY, reg_XY in zip(orig_XYs, reg_XYs):
            shape_type, _ = detect_and_regularize_shape(XY)
            
            # Print the name of the detected shape
            print(f"- {shape_type}")

            ax[0].plot(XY[:, 0], XY[:, 1], c=color, linewidth=2)
            ax[1].plot(reg_XY[:, 0], reg_XY[:, 1], c=color, linewidth=2)
            
            # Draw symmetry lines and count them
            draw_symmetry_lines(ax[1], shape_type, reg_XY, 'k')

    ax[0].set_title('Original Shapes')
    ax[1].set_title('Regularized Shapes with Symmetry Lines')

    plt.show()

def draw_symmetry_lines(ax, shape, reg_XY, color):
    def plot_line(ax, start, end, color):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle='--', linewidth=1)

    if shape == "Ellipse":
        x_min, y_min = np.min(reg_XY, axis=0)
        x_max, y_max = np.max(reg_XY, axis=0)
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        plot_line(ax, [x_min, center_y], [x_max, center_y], color)
        plot_line(ax, [center_x, y_min], [center_x, y_max], color)
        
        # Print number of symmetry lines
        print(f"  - {shape} has 2 lines of symmetry.")

    elif shape == "Rectangle":
        x_min, y_min = np.min(reg_XY, axis=0)
        x_max, y_max = np.max(reg_XY, axis=0)
        plot_line(ax, [(x_min + x_max)/2, y_min], [(x_min + x_max)/2, y_max], color)
        plot_line(ax, [x_min, (y_min + y_max)/2], [x_max, (y_min + y_max)/2], color)
        
        # Print number of symmetry lines
        print(f"  - {shape} has 2 lines of symmetry.")

    elif shape == "Star":
        num_points = len(reg_XY) // 2
        for i in range(num_points):
            start_point = reg_XY[i]
            end_point = reg_XY[(i + num_points) % len(reg_XY)]
            plot_line(ax, start_point, end_point, color)
        
        # Print number of symmetry lines
        print(f"  - {shape} has {num_points} lines of symmetry.")

    else:
        print(f"  - No specific symmetry lines defined for {shape}.")

def polylines2csv(paths_XYs, csv_path):
    """Writes the regularized shapes to a CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for path_XYs in paths_XYs:
            for i, XY in enumerate(path_XYs):
                for point in XY:
                    writer.writerow([i] + point.tolist())

def main():
    input_csv = "problems/isolated.csv"
    output_csv = "regularized_shapes.csv"
    path_XYs = read_csv(input_csv)
    regularized_XYs = []

    for path in path_XYs:
        reg_XYs = []
        for XY in path:
            _, reg_XY = detect_and_regularize_shape(XY)
            reg_XYs.append(reg_XY)
        regularized_XYs.append(reg_XYs)

    plot_shapes(path_XYs, regularized_XYs)
    polylines2csv(regularized_XYs, output_csv)

if __name__ == "__main__":
    main()