import streamlit as st
import cv2
import numpy as np
from scipy.spatial import Delaunay
import random
from PIL import Image

# Function to process triangles
def process_triangles(image, num_shapes):
    # Add padding to the image
    padding = 30
    image_padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)

    # Generate random points for Delaunay triangulation
    height, width, _ = image_padded.shape
    num_points = num_shapes + 2
    points = np.array([[random.randint(0, width), random.randint(0, height)] for _ in range(num_points)])
    triangles = Delaunay(points)

    # Create a copy of the image to draw triangles
    final_image = image.copy()

    for simplex in triangles.simplices:
        triangle_points = points[simplex]
        triangle_points_no_padding = triangle_points - [padding, padding]

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(triangle_points), 255)
        masked_image = cv2.bitwise_and(image_padded, image_padded, mask=mask)

        avg_color = cv2.mean(masked_image, mask=mask)[:3]
        avg_color_bgr = tuple(map(int, (avg_color[2], avg_color[1], avg_color[0])))

        alpha = 0.7
        overlay = final_image.copy()
        cv2.fillConvexPoly(overlay, np.int32(triangle_points_no_padding), avg_color_bgr)
        final_image = cv2.addWeighted(final_image, 1 - alpha, overlay, alpha, 0)

    return final_image


# Function to process rectangles
def process_rectangles(input_image, num_shapes):

    def generate_max_random_rectangles(image_size=(512, 512), min_size=(5, 5), max_size=(10, 10), max_attempts=500000, max_fail_attempts=10000, max_rectangles_limit=10):
        # Create a black image
        img = np.zeros(image_size, dtype=np.uint8)

        # Store rectangles to ensure no overlap
        rectangles = []

        def is_too_close(x, y, width, height):
            # Check if the rectangle is too close to any other rectangle
            for (rx, ry, rw, rh) in rectangles:
                if (rx < x + width and rx + rw > x and ry < y + height and ry + rh > y):
                    return True
            return False

        attempts = 0
        failed_attempts = 0

        while attempts < max_attempts and failed_attempts < max_fail_attempts and len(rectangles) < max_rectangles_limit:
            # Dynamically adjust width and height based on remaining space
            used_space = sum([rw * rh for (_, _, rw, rh) in rectangles])  # Area covered by existing rectangles
            total_space = image_size[0] * image_size[1]  # Total image space
            remaining_space = total_space - used_space  # Free space

            # Dynamically adjust size based on remaining space
            remaining_capacity = remaining_space / total_space
            min_dynamic_width = int(min_size[0] + (remaining_capacity * (max_size[0] - min_size[0])))
            min_dynamic_height = int(min_size[1] + (remaining_capacity * (max_size[1] - min_size[1])))
            max_dynamic_width = int(min_dynamic_width * 1.5)  # Max width can be larger based on remaining space
            max_dynamic_height = int(min_dynamic_height * 1.5)

            # Ensure dynamic size doesn't exceed initial max_size
            max_dynamic_width = min(max_dynamic_width, max_size[0])
            max_dynamic_height = min(max_dynamic_height, max_size[1])

            # Randomly choose a size within the new dynamic range
            width = random.randint(min_dynamic_width, max_dynamic_width)
            height = random.randint(min_dynamic_height, max_dynamic_height)

            # Random center position
            x = random.randint(0, image_size[1] - width)
            y = random.randint(0, image_size[0] - height)

            # Ensure rectangles don't overlap
            if not is_too_close(x, y, width, height):
                # Draw rectangle with just the border (outline)
                cv2.rectangle(img, (x, y), (x + width, y + height), 255, 1)  # 255 for white color, 1 for outline thickness
                rectangles.append((x, y, width, height))
                failed_attempts = 0  # Reset failed attempts since this one was successful
            else:
                failed_attempts += 1  # Increment failed attempts if rectangle couldn't be placed

            attempts += 1

        # Invert the image (255 - img)
        inverted_img = 255 - img

        return inverted_img, len(rectangles), rectangles


    def resize_image_to_shape(image, target_shape):
        # Resize the image to match the target shape
        return cv2.resize(image, (target_shape[1], target_shape[0]))


    def compute_average_under_rectangles(input_image, rect_image, rectangles):
        # Create an output image with an alpha channel (4 channels: BGRA)
        output_image = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)

        # For each rectangle, calculate the average pixel value from the input image
        for (x, y, width, height) in rectangles:
            # Create a mask for the rectangle
            mask = np.zeros(rect_image.shape, dtype=np.uint8)
            mask[y:y+height, x:x+width] = 255

            # Get the pixel values inside the rectangle from the input image
            rect_pixels = input_image[mask == 255]

            # Calculate the average pixel value
            if len(rect_pixels) > 0:
                average_value = np.mean(rect_pixels, axis=0)  # Take average across channels for color images
            else:
                average_value = [0, 0, 0]  # Default to black if no pixels

            # Set the average value to the rectangle area in the output image
            output_image[mask == 255] = np.append(average_value, [255])  # Add full opacity (alpha=255)

        return output_image


    def overlay_mask_on_image(input_image, mask_image):
        # Overlay the mask on the original image, using the alpha channel for transparency
        output_image = input_image.copy()

        # Where the mask has non-zero alpha, replace the original image with the mask's content
        mask_alpha = mask_image[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]

        # Blend input image and mask based on alpha
        for c in range(3):  # Loop through RGB channels
            output_image[:, :, c] = (1 - mask_alpha) * input_image[:, :, c] + mask_alpha * mask_image[:, :, c]

        return output_image


    # Load your input image (replace 'input_image.jpg' with your image path)
    #input_image = cv2.imread('test44.jpg')  # For color image
    input_image = cv2.resize(input_image, (256, 256))

    # Get the shape of the input image
    input_shape = input_image.shape

    # Generate the rectangle image and get its number of rectangles
    rect_image, num_rectangles, rectangles = generate_max_random_rectangles(image_size=(input_shape[0], input_shape[1]), max_rectangles_limit=num_shapes)

    # Resize the rectangle image to match the input image's shape
    resized_rect_image = resize_image_to_shape(rect_image, input_shape[:2])

    # Compute the average pixel value under each rectangle
    averaged_rect_image = compute_average_under_rectangles(input_image, resized_rect_image, rectangles)

    # Overlay the mask with averaged values on the original input image
    overlayed_image = overlay_mask_on_image(input_image, averaged_rect_image)

    # Save the overlayed image to a file
    #cv2.imwrite('overlayed_image.png', overlayed_image)

    # Print the number of rectangles placed
    # print(f"Number of rectangles placed: {num_rectangles}")
    return overlayed_image ,num_rectangles




# Function to process circles
def process_circles(input_image, num_shapes):

    def generate_max_random_circles(image_size=(512, 512), min_radius=0.1, max_radius=5, max_attempts=50000, max_fail_attempts=10000, max_circles_limit=10):
        # Create a black image
        img = np.zeros(image_size, dtype=np.uint8)

        # Store circles to ensure no overlap
        circles = []

        def is_too_close(x, y, radius):
            # Check if the circle is too close to any other circle
            for (cx, cy, cr) in circles:
                distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                if distance < (cr + radius):  # If the circles overlap
                    return True
            return False

        attempts = 0
        failed_attempts = 0

        while attempts < max_attempts and failed_attempts < max_fail_attempts and len(circles) < max_circles_limit:
            # Dynamically adjust radius based on remaining space
            used_space = sum([np.pi * cr**2 for (_, _, cr) in circles])  # Area covered by existing circles
            total_space = image_size[0] * image_size[1]  # Total image space
            remaining_space = total_space - used_space  # Free space

            # Dynamically adjust min_radius and max_radius based on remaining space
            remaining_capacity = remaining_space / total_space
            min_dynamic_radius = int(min_radius + (remaining_capacity * (max_radius - min_radius)))
            max_dynamic_radius = int(min_dynamic_radius * 1.5)  # Max radius can be larger based on remaining space

            # Ensure dynamic radius doesn't exceed initial max_radius
            max_dynamic_radius = min(max_dynamic_radius, max_radius)

            # Randomly choose a radius within the new dynamic range
            radius = random.randint(min_dynamic_radius, max_dynamic_radius)

            # Random center position
            center_x = random.randint(radius, image_size[1] - radius)
            center_y = random.randint(radius, image_size[0] - radius)

            # Ensure circles don't overlap
            if not is_too_close(center_x, center_y, radius):
                # Draw circle with just the border (outline)
                cv2.circle(img, (center_x, center_y), radius, 255, 1)  # 255 for white color, 1 for outline thickness
                circles.append((center_x, center_y, radius))
                failed_attempts = 0  # Reset failed attempts since this one was successful
            else:
                failed_attempts += 1  # Increment failed attempts if circle couldn't be placed

            attempts += 1

        # Invert the image (255 - img)
        inverted_img = 255 - img

        return inverted_img, len(circles), circles


    def resize_image_to_shape(image, target_shape):
        # Resize the image to match the target shape
        return cv2.resize(image, (target_shape[1], target_shape[0]))


    def compute_average_under_circles(input_image, circle_image, circles):
        # Create an output image with an alpha channel (4 channels: BGRA)
        output_image = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)

        # Resize the circle image to match the input image
        resized_circle_image = cv2.resize(circle_image, (input_image.shape[1], input_image.shape[0]))

        # For each circle, calculate the average pixel value from the input image
        for (cx, cy, radius) in circles:
            # Create a mask for the circle
            y, x = np.ogrid[:resized_circle_image.shape[0], :resized_circle_image.shape[1]]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2

            # Get the pixel values inside the circle from the input image
            circle_pixels = input_image[mask]

            # Calculate the average pixel value
            if len(circle_pixels) > 0:
                average_value = np.mean(circle_pixels, axis=0)  # Take average across channels for color images
            else:
                average_value = [0, 0, 0]  # Default to black if no pixels

            # Set the average value to the circle area in the output image
            output_image[mask] = np.append(average_value, [255])  # Add full opacity (alpha=255)

        return output_image


    def overlay_mask_on_image(input_image, mask_image):
        # Overlay the mask on the original image, using the alpha channel for transparency
        output_image = input_image.copy()

        # Where the mask has non-zero alpha, replace the original image with the mask's content
        mask_alpha = mask_image[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]

        # Blend input image and mask based on alpha
        for c in range(3):  # Loop through RGB channels
            output_image[:, :, c] = (1 - mask_alpha) * input_image[:, :, c] + mask_alpha * mask_image[:, :, c]

        return output_image


    # Load your input image (replace 'input_image.jpg' with your image path)
    # input_image = cv2.imread('test44.jpg')  # For color image
    input_image = cv2.resize(input_image, (256, 256))

    # Get the shape of the input image
    h, w, _ = input_image.shape

    # Generate the circle image and get its number of circles
    circle_image, num_circles, circles = generate_max_random_circles(image_size=(h, w), max_circles_limit=num_shapes)

    # Resize the circle image to match the input image's shape
    resized_circle_image = resize_image_to_shape(circle_image, input_image.shape[:2])

    # Compute the average pixel value under each circle
    averaged_circle_image = compute_average_under_circles(input_image, resized_circle_image, circles)

    # Overlay the mask with averaged values on the original input image
    overlayed_image = overlay_mask_on_image(input_image, averaged_circle_image)

    # Save the final image to disk
    # cv2.imwrite('overlayed_image.jpg', overlayed_image)

    # Print the number of circles placed
    # print(f"Number of circles placed: {num_circles}")
    return overlayed_image ,num_circles
def page_1():
    st.title("Image Overlay with Shapes")
    st.write("Upload an image, choose a shape, and customize the overlay.")

    # Image upload
    uploaded_file = st.file_uploader("Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Load and process the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Error: Image not found.")
        else:
            # Resize image for easier processing
            image_resized = cv2.resize(image, (500, 500))

            # Select shape
            shape = st.selectbox("Choose a shape:", ["None", "Triangle", "Rectangle", "Circle"])

            if shape != "None":
                # Number of shapes (default is 50)
                num_shapes = st.number_input(f"Number of {shape}s", min_value=1, step=1, value=50)

                if shape == "Triangle":
                    final_image = process_triangles(image_resized, num_shapes)
                    nu =num_shapes
                elif shape == "Rectangle":
                    final_image , nu = process_rectangles(image_resized, num_shapes)
                elif shape == "Circle":
                    final_image, nu= process_circles(image_resized, num_shapes)

                # Convert the image to RGB for display
                final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

                # Create two columns layout
                col1, col2 = st.columns(2)

                # Display the original and final image in columns
                with col1:
                    st.image(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

                with col2:
                    st.image(final_image_rgb, caption=f"Image with {nu} {shape}s ", use_column_width=True)

                # Convert final image back to BGR for downloading
                final_image_bgr = cv2.cvtColor(final_image_rgb, cv2.COLOR_RGB2BGR)

                # Save the final image to a temporary file
                _, img_buffer = cv2.imencode('.png', final_image_bgr)
                img_bytes = img_buffer.tobytes()

                # Provide download button
                st.download_button(
                    label="Download Output Image",
                    data=img_bytes,
                    file_name="output_image.png",
                    mime="image/png"
                )
def page_2():
    import cv2
    import numpy as np
    import streamlit as st
    from PIL import Image
    import io
    import webcolors

    # Function to calculate the center of a contour
    def get_center(contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)

    # Function to check overlap between triangles
    def is_overlapping(center, existing_centers):
        max_overlap_dist = 20
        for ec in existing_centers:
            if np.linalg.norm(np.array(center) - np.array(ec)) < max_overlap_dist:
                return True
        return False

    # Function to handle dynamic categories with angles, tolerances, and RGB selection
    def handle_category(index):
        st.subheader(f"Category {index + 1}")
        
        # Angles and tolerances in one line using columns
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        
        angles = []
        
        with col1:
            angle_1 = st.number_input(f"Angle 1 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_1')
        with col2:
            tolerance_1 = st.number_input(f"Tolerance 1 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_1')
            
        with col3:
            angle_2 = st.number_input(f"Angle 2 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_2')
        with col4:
            tolerance_2 = st.number_input(f"Tolerance 2 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_2')
            
        with col5:
            angle_3 = st.number_input(f"Angle 3 (°)", value=60.0, step=0.1, min_value=0.0, key=f'angle_{index}_3')
        with col6:
            tolerance_3 = st.number_input(f"Tolerance 3 (°)", value=1.0, step=0.1, key=f'tolerance_{index}_3')

        # Save the angles and tolerances into a list of tuples
        angles = [(angle_1, tolerance_1), (angle_2, tolerance_2), (angle_3, tolerance_3)]
        
        # RGB Color Selection
        with col7:
            hex_color = st.color_picker(f"Select RGB Color for Category {index + 1}", key=f'rgb_color_{index}')
        
        # Convert hex to RGB
        try:
            rgb_color = webcolors.hex_to_rgb(hex_color)
        except ValueError:
            rgb_color = (255, 0, 0)  # Default to red if there's an issue with the hex color

        # Checkbox for strict matching (all angles or any angle)
        match_all_angles = st.checkbox(f"Match all angles?", value=True, key=f"checkbox_{index}")

        # Check if the sum of angles is equal to 180
        if sum([angle for angle, _ in angles]) != 180:
            st.warning("The sum of the angles must be exactly 180°. Please adjust the angles.")
        
        return angles, rgb_color, match_all_angles

    # Function to process the uploaded image and detect triangles
    def process_image(image_path):
        # Load image
        image = cv2.imread(image_path)

        # Convert to grayscale and apply binary threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Erosion and dilation
        kernel_size = (3, 3)
        kernel = np.ones(kernel_size, np.uint8)

        eroded = cv2.erode(th, kernel, iterations=3)
        dilated = cv2.dilate(th, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Copy image for drawing results
        image_copy = image.copy()

        # Set thresholds for filtering
        min_area = 500  # Minimum area of triangles to draw
        drawn_centers = []

        # Process contours from eroded image
        for contour in contours:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3 and cv2.contourArea(approx) > min_area:
                center = get_center(approx)
                if center and not is_overlapping(center, drawn_centers):
                    drawn_centers.append(center)
                    cv2.drawContours(image_copy, [approx], -1, (0, 0, 255), 3)  # Default red for triangles

        # Process contours from thresholded image
        for contour in contours1:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 3 and cv2.contourArea(approx) > min_area:
                center = get_center(approx)
                if center and not is_overlapping(center, drawn_centers):
                    drawn_centers.append(center)
                    cv2.drawContours(image_copy, [approx], -1, (0, 0, 255), 3)  # Default red for triangles

        return image_copy, contours, contours1

    # Main function to organize the layout

    st.title("Image Triangle Detection and Category Management")

    # Image upload
    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Load and process the uploaded image
        image = Image.open(uploaded_file)
        img_path = f"temp_image.jpg"
        image.save(img_path)

        # Process the image and get contours and the output image with triangles
        image_with_triangles, contours, contours1 = process_image(img_path)

        # Show original and processed images side by side
        
        # Category management section
        if 'categories' not in st.session_state:
            st.session_state.categories = []  # Initialize categories as an empty list

        # Create or delete categories
        if st.button("Add Category"):
            st.session_state.categories.append({
                'angles': [],
                'rgb_color': (255, 0, 0),  # Default color red
                'match_all_angles': True
            })

        # Delete Category
        category_to_delete = st.selectbox("Select Category to Delete", [""] + [f"Category {i+1}" for i in range(len(st.session_state.categories))], key="category_to_delete")
        
        if category_to_delete:
            category_index = int(category_to_delete.split()[-1]) - 1
            if st.button(f"Delete {category_to_delete}"):
                del st.session_state.categories[category_index]
                st.session_state.categories = st.session_state.categories  # Refresh the list after deletion
        
        # Display and handle each category
        for index, category in enumerate(st.session_state.categories):
            angles, rgb_color, match_all_angles = handle_category(index)
            st.session_state.categories[index]['angles'] = angles
            st.session_state.categories[index]['rgb_color'] = rgb_color
            st.session_state.categories[index]['match_all_angles'] = match_all_angles

            # Apply color to triangles based on category
            for contour in contours:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx) == 3 and cv2.contourArea(approx) > 500:
                    triangle_angles = [60.0, 60.0, 60.0]  # Dummy values for angles, calculate using actual method
                    angle_match = False
                    if match_all_angles:
                        if all(angle - tolerance <= triangle_angle <= angle + tolerance for triangle_angle, (angle, tolerance) in zip(triangle_angles, angles)):
                            angle_match = True
                    else:
                        if any(angle - tolerance <= triangle_angle <= angle + tolerance for triangle_angle, (angle, tolerance) in zip(triangle_angles, angles)):
                            angle_match = True
                    if angle_match:
                        cv2.drawContours(image_with_triangles, [approx], -1, rgb_color, 3)

        # Provide download functionality for the processed image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(image_with_triangles, caption="Detected Triangles", use_column_width=True)
        image_with_triangles= cv2.cvtColor(image_with_triangles, cv2
                                        .COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', image_with_triangles)
        img_bytes = buffer.tobytes()
        
        
        st.download_button( 
            label="Download Processed Image",
            data=img_bytes,
            file_name="processed_image.png",
            mime="image/png"
        )



    # Add your code for the second page here

# Function to display the menu and navigate between pages

menu = ["Image Generation", "Triangle Detection"]
choice = st.sidebar.radio("Select an Option", menu)

if choice == "Image Generation":
    page_1()
elif choice == "Triangle Detection":
    page_2()

