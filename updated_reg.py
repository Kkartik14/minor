import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
from rdp import rdp
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares
import sys
import os


def read_csv(csv_path):
    """Reads a CSV file and organizes the points into paths."""
    try:
        np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    except IOError:
        print(f"Error: The file {csv_path} could not be read.")
        sys.exit(1)

    path_XYs = []
    for path_id in np.unique(np_path_XYs[:, 0]):
        path_df = np_path_XYs[np_path_XYs[:, 0] == path_id]
        subpaths = []
        for subpath_id in np.unique(path_df[:, 1]):
            subpath_df = path_df[path_df[:, 1] == subpath_id]
            XY = subpath_df[:, 2:]
            if XY.shape[0] < 2:
                continue  # Ignore subpaths with insufficient points
            subpaths.append(XY)
        path_XYs.append(subpaths)
    print(f"Total paths read: {len(path_XYs)}")
    return path_XYs


def polylines2csv(paths_XYs, shapes, csv_path):
    """Writes the regularized shapes and their types to a CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['PathID', 'SubpathID', 'ShapeType', 'X', 'Y'])
        for path_id, (path_XYs, shape_types) in enumerate(zip(paths_XYs, shapes)):
            for subpath_id, (reg_XY, shape_type) in enumerate(zip(path_XYs, shape_types)):
                for point in reg_XY:
                    writer.writerow([path_id, subpath_id, shape_type] + point.tolist())


def simplify_contour(XY, epsilon=0.3):
    """
    Simplifies the contour using the RDP algorithm.

    Parameters:
        XY (ndarray): Array of XY coordinates.
        epsilon (float): Distance threshold for simplification.

    Returns:
        simplified_XY (ndarray): Simplified contour.
    """
    simplified_XY = rdp(XY, epsilon=epsilon)
    print(f"Simplified contour from {len(XY)} to {len(simplified_XY)} points.")
    return simplified_XY


def smooth_contour(XY, s=2.0):
    """
    Smooths the contour using B-Splines.

    Parameters:
        XY (ndarray): Array of XY coordinates.
        s (float): Smoothing factor.

    Returns:
        smooth_XY (ndarray): Smoothed contour.
    """
    try:
        tck, _ = splprep([XY[:, 0], XY[:, 1]], s=s)
        u = np.linspace(0, 1, max(len(XY), 100))
        smooth_X, smooth_Y = splev(u, tck)
        smooth_XY = np.vstack((smooth_X, smooth_Y)).T
        print(f"Smoothed contour to {len(smooth_XY)} points.")
        return smooth_XY
    except Exception as e:
        print(f"Error in smoothing contour: {e}")
        return XY  # Return original if smoothing fails


def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    cb = b - c
    norm_ab = np.linalg.norm(ab) + 1e-8
    norm_cb = np.linalg.norm(cb) + 1e-8
    cosine_angle = np.dot(ab, cb) / (norm_ab * norm_cb)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


def is_straight_line(XY, tolerance=0.02):
    """Checks if the points form a straight line."""
    if len(XY) < 3:
        return True

    # Fit a line using linear regression
    A = np.vstack([XY[:, 0], np.ones(len(XY))]).T
    m, c = np.linalg.lstsq(A, XY[:, 1], rcond=None)[0]
    # Calculate residuals
    residuals = XY[:, 1] - (m * XY[:, 0] + c)
    rms = np.sqrt(np.mean(residuals ** 2))
    # Normalize RMS by the bounding box height
    bbox_height = np.max(XY[:, 1]) - np.min(XY[:, 1]) + 1e-8
    normalized_rms = rms / bbox_height
    print(f"Normalized RMS for straight line check: {normalized_rms:.3f}")
    return normalized_rms < tolerance


def is_square(contour, tolerance=0.05):
    """Checks if the contour forms a square."""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    width = np.linalg.norm(box[0] - box[1])
    height = np.linalg.norm(box[1] - box[2])
    aspect_ratio = min(width, height) / (max(width, height) + 1e-8)

    contour_area = cv2.contourArea(contour)
    rect_area = width * height
    area_ratio = contour_area / rect_area

    is_square_shape = abs(1 - aspect_ratio) < tolerance
    is_filled = area_ratio > (1 - tolerance)

    print(f"Square check - Aspect Ratio: {aspect_ratio:.3f}, Area Ratio: {area_ratio:.3f}")
    return is_square_shape and is_filled


def is_star(approx, tolerance=0.3):
    """Checks if the contour forms a star shape."""
    num_vertices = len(approx)
    # Allow a range of vertices for flexibility, typically 8-12 for a 5-point star
    if not (8 <= num_vertices <= 12):
        print(f"Star check failed: Number of vertices {num_vertices} not in range 8-12.")
        return False

    # Calculate angles at each vertex
    angles = [
        calculate_angle(
            approx[i][0],
            approx[(i + 1) % num_vertices][0],
            approx[(i + 2) % num_vertices][0]
        )
        for i in range(num_vertices)
    ]

    # Count sharp angles (e.g., <60 degrees or >300 degrees)
    sharp_angles = [angle for angle in angles if angle < 60 or angle > 300]

    if len(sharp_angles) >= 4:  # Reduced from 5 for better detection
        print(f"Star check passed: {num_vertices} vertices, {len(sharp_angles)} sharp angles.")
        return True
    else:
        print(f"Star check failed: Not enough sharp angles ({len(sharp_angles)} found).")
        return False


def detect_shape(XY):
    """
    Detects the shape type from the provided points.

    Parameters:
        XY (ndarray): Array of XY coordinates.

    Returns:
        shape (str): Detected shape type.
    """
    contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)  # Adjusted epsilon for better flexibility
    area = cv2.contourArea(contour)
    print(f"\nDetecting shape:")
    print(f"Perimeter: {peri:.2f}, Approx Vertices: {len(approx)}, Area: {area:.2f}")

    if area < 100:
        print("Shape too small to detect.")
        return "Unknown"

    if is_straight_line(XY):
        print("Detected shape: Straight Line")
        return "Straight Line"

    if len(approx) == 4:
        if is_square(contour):
            print("Detected shape: Square")
            return "Square"
        else:
            print("Detected shape: Rectangle")
            return "Rectangle"

    if 8 <= len(approx) <= 12:
        if is_star(approx):
            print("Detected shape: Star")
            return "Star"

    # Additional condition for circles and ellipses
    circularity = 4 * np.pi * area / (peri * peri + 1e-8)
    print(f"Circularity: {circularity:.2f}")
    if circularity > 0.90:
        print("Detected shape: Circle")
        return "Circle"
    elif circularity > 0.75:
        print("Detected shape: Ellipse")
        return "Ellipse"

    print("Shape not recognized.")
    return "Unknown"


def lm_circle_fitting(XY, initial_guess=(0, 0, 1)):
    """
    Fits a circle to the points using the Levenberg-Marquardt algorithm.

    Parameters:
        XY (ndarray): Array of XY coordinates.
        initial_guess (tuple): Initial guess for (xc, yc, r).

    Returns:
        xc (float): X-coordinate of circle center.
        yc (float): Y-coordinate of circle center.
        r (float): Radius of the circle.
    """
    def residuals(params, x, y):
        xc, yc, r = params
        return np.sqrt((x - xc)**2 + (y - yc)**2) - r

    x = XY[:, 0]
    y = XY[:, 1]
    try:
        result = least_squares(residuals, initial_guess, args=(x, y), method='lm')
        xc, yc, r = result.x
        print(f"LM Circle Fitting: Center=({xc:.2f}, {yc:.2f}), Radius={r:.2f}")
        return xc, yc, r
    except Exception as e:
        print(f"LM Circle Fitting failed: {e}")
        return np.mean(XY[:, 0]), np.mean(XY[:, 1]), 1  # Fallback values


def regularize_circle(XY):
    """
    Regularizes the points to form a perfect circle.

    Parameters:
        XY (ndarray): Array of XY coordinates.

    Returns:
        regularized_XY (ndarray): Regularized circle coordinates.
    """
    xc, yc, r = lm_circle_fitting(XY)
    t = np.linspace(0, 2 * np.pi, len(XY))
    x_fit = xc + r * np.cos(t)
    y_fit = yc + r * np.sin(t)
    regularized_XY = np.vstack((x_fit, y_fit)).T
    print("Regularized to Circle.")
    return regularized_XY


def regularize_ellipse(XY):
    """Regularizes the points to form a perfect ellipse."""
    try:
        contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
        ellipse = cv2.fitEllipse(contour)
        center, axes, angle = ellipse
        a, b = axes  # Major and minor axes lengths
        theta = np.deg2rad(angle)
        t = np.linspace(0, 2 * np.pi, len(XY))

        # Parametric equation of ellipse
        x_fit = center[0] + (a / 2) * np.cos(t) * np.cos(theta) - (b / 2) * np.sin(t) * np.sin(theta)
        y_fit = center[1] + (a / 2) * np.cos(t) * np.sin(theta) + (b / 2) * np.sin(t) * np.cos(theta)
        regularized_XY = np.vstack((x_fit, y_fit)).T
        print("Regularized to Ellipse.")
        return regularized_XY
    except Exception as e:
        print(f"Ellipse fitting failed: {e}")
        return XY  # Return original if fitting fails


def regularize_rectangle(XY, closed=True):
    """Regularizes the points to form a perfect rectangle."""
    try:
        contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Order the points in a consistent order (clockwise starting from top-left)
        box = order_box_points(box)

        # Close the rectangle if it's a closed path
        if closed:
            regularized_XY = np.vstack((box, box[0]))
        else:
            regularized_XY = box

        print("Regularized to Rectangle.")
        return regularized_XY, rect  # Return rect for orientation
    except Exception as e:
        print(f"Rectangle fitting failed: {e}")
        return XY, None  # Return original if fitting fails


def regularize_square(XY, closed=True):
    """Regularizes the points to form a perfect square."""
    try:
        contour = np.array(XY, dtype=np.int32).reshape((-1, 1, 2))
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Order the points in a consistent order (clockwise starting from top-left)
        box = order_box_points(box)

        # Calculate side lengths and adjust to make all sides equal (average side length)
        side_lengths = [
            np.linalg.norm(box[0] - box[1]),
            np.linalg.norm(box[1] - box[2]),
            np.linalg.norm(box[2] - box[3]),
            np.linalg.norm(box[3] - box[0])
        ]
        avg_length = np.mean(side_lengths)

        # Calculate direction vectors
        vectors = [
            (box[1] - box[0]) / (side_lengths[0] + 1e-8),
            (box[2] - box[1]) / (side_lengths[1] + 1e-8),
            (box[3] - box[2]) / (side_lengths[2] + 1e-8),
            (box[0] - box[3]) / (side_lengths[3] + 1e-8)
        ]

        # Adjust points based on average side length
        new_box = []
        for i in range(4):
            new_point = box[i] + vectors[i] * (avg_length - side_lengths[i]) / 2
            new_box.append(new_point)

        new_box = np.array(new_box, dtype=np.float32)

        # Re-order and close the square if necessary
        new_box = order_box_points(new_box)
        if closed:
            regularized_XY = np.vstack((new_box, new_box[0]))
        else:
            regularized_XY = new_box

        print("Regularized to Square.")
        return regularized_XY, rect  # Return rect for orientation
    except Exception as e:
        print(f"Square fitting failed: {e}")
        return XY, None  # Return original if fitting fails


def regularize_star(XY, num_points=5):
    """Regularizes the points to form a perfect star."""
    centroid = np.mean(XY, axis=0)

    # Compute bounding box dimensions
    x_min, x_max = np.min(XY[:, 0]), np.max(XY[:, 0])
    y_min, y_max = np.min(XY[:, 1]), np.max(XY[:, 1])

    # Determine the radii based on bounding box
    outer_radius = np.max(np.linalg.norm(XY - centroid, axis=1))
    inner_radius = outer_radius * 0.5  # Adjust inner radius as needed

    angles = np.linspace(0, 2 * np.pi, 2 * num_points, endpoint=False)

    star_points = []
    for i, angle in enumerate(angles):
        r = outer_radius if i % 2 == 0 else inner_radius
        x = centroid[0] + r * np.cos(angle)
        y = centroid[1] + r * np.sin(angle)
        star_points.append([x, y])

    star_points.append(star_points[0])  # Close the star

    return np.array(star_points)


def regularize_straight_line(XY, closed=True):
    """Regularizes the points to form a perfect straight line."""
    if not closed:
        return np.array([XY[0], XY[-1]])
    else:
        print("Regularize_straight_line is intended for open paths. Keeping original shape.")
        return XY


def order_box_points(box):
    """
    Orders the box points in clockwise order starting from the top-left.

    Parameters:
        box (ndarray): Array of four points.

    Returns:
        ordered_box (ndarray): Ordered array of four points.
    """
    # Initialize a list to hold the ordered points
    ordered_box = np.zeros((4, 2), dtype="float32")

    # Sum and difference of points for ordering
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1)

    ordered_box[0] = box[np.argmin(s)]      # Top-left has the smallest sum
    ordered_box[2] = box[np.argmax(s)]      # Bottom-right has the largest sum
    ordered_box[1] = box[np.argmin(diff)]   # Top-right has the smallest difference
    ordered_box[3] = box[np.argmax(diff)]   # Bottom-left has the largest difference

    return ordered_box


def calculate_symmetry_lines(XY, shape_type, rect=None):
    """Calculates symmetry lines for the given shape."""
    symmetry_lines = []
    centroid = np.mean(XY, axis=0)

    x_min, x_max = np.min(XY[:, 0]), np.max(XY[:, 0])
    y_min, y_max = np.min(XY[:, 1]), np.max(XY[:, 1])

    if shape_type in ["Circle", "Square", "Rectangle"]:
        if shape_type in ["Square", "Rectangle"] and rect is not None:
            # Use the orientation from minAreaRect
            angle = rect[2]
            theta = np.deg2rad(angle)

            # Compute unit vectors for major and minor axes
            major_axis = np.array([np.cos(theta), np.sin(theta)])
            minor_axis = np.array([-np.sin(theta), np.cos(theta)])

            # Determine the length of symmetry lines based on rectangle dimensions
            width, height = rect[1]
            half_width = width / 2
            half_height = height / 2

            # Define symmetry lines along major axis
            line1_start = centroid - major_axis * half_width
            line1_end = centroid + major_axis * half_width
            symmetry_lines.append( (line1_start.tolist(), line1_end.tolist()) )

            # Define symmetry lines along minor axis
            line2_start = centroid - minor_axis * half_height
            line2_end = centroid + minor_axis * half_height
            symmetry_lines.append( (line2_start.tolist(), line2_end.tolist()) )
        else:
            # For circles, simple vertical and horizontal lines
            vertical = ([centroid[0], y_min], [centroid[0], y_max])
            horizontal = ([x_min, centroid[1]], [x_max, centroid[1]])
            symmetry_lines.extend([vertical, horizontal])
    elif shape_type == "Ellipse":
        # Only horizontal symmetry line
        horizontal = ([x_min, centroid[1]], [x_max, centroid[1]])
        symmetry_lines.append(horizontal)
    elif shape_type == "Star":
        # Identify outer points based on distance from centroid
        distances = np.linalg.norm(XY - centroid, axis=1)
        outer_radius = np.max(distances)
        epsilon = outer_radius * 0.05  # 5% tolerance

        # Select outer points
        outer_points = XY[np.abs(distances - outer_radius) < epsilon]

        # Draw symmetry lines from centroid to each outer point
        for outer_point in outer_points:
            symmetry_lines.append( (centroid.tolist(), outer_point.tolist()) )
    elif shape_type == "Straight Line":
        # The line itself is the symmetry line
        symmetry_lines.append( ([XY[0][0], XY[0][1]], [XY[-1][0], XY[-1][1]]) )

    return symmetry_lines


def detect_and_regularize_shape(XY, closed=True, rdp_epsilon=0.3, spline_s=1.0):
    """
    Detects the shape and returns the regularized points after simplification and smoothing.

    Parameters:
        XY (ndarray): Array of XY coordinates.
        closed (bool): Indicates whether the path is closed.
        rdp_epsilon (float): Epsilon parameter for RDP.
        spline_s (float): Smoothing factor for B-Splines.

    Returns:
        shape (str): Detected shape type.
        regularized_XY (ndarray): Regularized XY coordinates.
        rect (tuple or None): Rectangle parameters if shape is Rectangle or Square.
    """
    simplified_XY = simplify_contour(XY, epsilon=rdp_epsilon)
    smooth_XY = smooth_contour(simplified_XY, s=spline_s)

    shape = detect_shape(smooth_XY)

    rect = None  # Initialize rect

    if shape == "Circle":
        regularized_XY = regularize_circle(smooth_XY)
    elif shape == "Ellipse":
        regularized_XY = regularize_ellipse(smooth_XY)
    elif shape == "Rectangle":
        regularized_XY, rect = regularize_rectangle(smooth_XY, closed)
    elif shape == "Square":
        regularized_XY, rect = regularize_square(smooth_XY, closed)
    elif shape == "Star":
        regularized_XY = regularize_star(smooth_XY)
    elif shape == "Straight Line":
        regularized_XY = regularize_straight_line(smooth_XY, closed)
    else:
        regularized_XY = smooth_XY  # If unknown, keep the smoothed contour

    return shape, regularized_XY, rect


def plot_shapes(path_XYs, regularized_XYs, shapes, rects):
    """Plots original and regularized shapes with symmetry lines, and prints information about detected shapes."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax_orig, ax_reg = axes
    ax_orig.set_title('Original Shapes')
    ax_reg.set_title('Regularized Shapes with Symmetry Lines')

    colors = plt.cm.tab20.colors  # Updated to avoid deprecation warning

    print("\033[1mShapes Detected and Regularized:\033[0m")

    for path_id, (orig_subpaths, reg_subpaths, shape_types, rect_list) in enumerate(zip(path_XYs, regularized_XYs, shapes, rects)):
        color = colors[path_id % len(colors)]
        for subpath_id, (orig_XY, reg_XY, shape_type, rect) in enumerate(zip(orig_subpaths, reg_subpaths, shape_types, rect_list)):
            print(f"Path {path_id}, Subpath {subpath_id}: {shape_type}")

            # Plot original shape
            ax_orig.plot(orig_XY[:, 0], orig_XY[:, 1], color=color, linewidth=1.5,
                        label=f'Path {path_id}' if subpath_id == 0 else "")

            # Plot regularized shape
            ax_reg.plot(reg_XY[:, 0], reg_XY[:, 1], color=color, linewidth=2,
                       label=f'Path {path_id}' if subpath_id == 0 else "")

            # Calculate and plot symmetry lines
            symmetry_lines = calculate_symmetry_lines(reg_XY, shape_type, rect=rect if shape_type in ["Rectangle", "Square"] else None)
            for line in symmetry_lines:
                (x_start, y_start), (x_end, y_end) = line
                ax_reg.plot([x_start, x_end], [y_start, y_end], color='k',
                           linestyle='--', linewidth=1)

    ax_orig.set_aspect('equal', adjustable='datalim')
    ax_reg.set_aspect('equal', adjustable='datalim')

    # Handle legends
    handles_orig, labels_orig = ax_orig.get_legend_handles_labels()
    by_label_orig = dict(zip(labels_orig, handles_orig))
    ax_orig.legend(by_label_orig.values(), by_label_orig.keys())

    handles_reg, labels_reg = ax_reg.get_legend_handles_labels()
    by_label_reg = dict(zip(labels_reg, handles_reg))
    ax_reg.legend(by_label_reg.values(), by_label_reg.keys())

    plt.show()


def main():
    """
    Main function to execute the shape regularization and symmetry detection workflow.
    """
    # Configuration Parameters
    input_csv = "problems/isolated.csv"        # Input CSV path
    output_csv = "regularized_shapes.csv"      # Output CSV path
    rdp_epsilon = 0.3                           # RDP simplification parameter (adjusted for less simplification)
    spline_s = 1.0                              # B-Spline smoothing factor

    if not os.path.exists(input_csv):
        print(f"Error: Input file {input_csv} does not exist.")
        sys.exit(1)

    path_XYs = read_csv(input_csv)

    regularized_XYs = []
    shapes_detected = []
    rects = []  # To store rectangle parameters for symmetry lines

    for path_id, path in enumerate(path_XYs):
        reg_path = []
        shape_types = []
        rect_list = []
        for subpath_id, XY in enumerate(path):
            is_closed = np.linalg.norm(XY[0] - XY[-1]) < 5.0  # Threshold to determine if path is closed
            print(f"\nProcessing Path {path_id}, Subpath {subpath_id} - Closed: {is_closed}")
            shape, reg_XY, rect = detect_and_regularize_shape(XY, closed=is_closed,
                                                            rdp_epsilon=rdp_epsilon,
                                                            spline_s=spline_s)
            reg_path.append(reg_XY)
            shape_types.append(shape)
            rect_list.append(rect)
        regularized_XYs.append(reg_path)
        shapes_detected.append(shape_types)
        rects.append(rect_list)
    plot_shapes(path_XYs, regularized_XYs, shapes_detected, rects)
    polylines2csv(regularized_XYs, shapes_detected, output_csv)
    print(f"\nRegularized shapes have been saved to {output_csv}")


if __name__ == "__main__":
    main()