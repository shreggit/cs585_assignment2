import numpy as np
import cv2
import os
import re

def detect_skin(frame):
    #define the lower and upper bound for skin colors
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    #convert the frame to HSV colors
    converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #determine the regions of the converted frame in the skin color region
    skin_mask = cv2.inRange(converted_frame, lower, upper)
    #using a elliptical kernel, apply a series of erosion and dilations to the skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    #blur the skin mask and apply bitwise with itself and pass it through the skin mask
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skin_mask)
    return skin

def convert_to_binary(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_frame, (35, 35), 0)
    _, binary_frame = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary_frame

def background_difference(background, frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(gray_frame, background)
    _, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
    return thresh

def frame_difference(prev_frame, current_frame):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(prev_frame_gray, current_frame_gray)
    _, thresh = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)
    return thresh

def calculate_circularity(contour):
    try:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (perimeter) ** 2
    except:
        circularity = 0
    return circularity

def load_templates():
    templates = []
    x, y, w, h = 50, 50, 500, 500
    templates = {}
    temp_dir = "templates"
    for file in os.listdir(temp_dir):
        match = re.match(r'(open_palms|closed_fist|index|thumbs_up|thumbs_down).png', file)
        if match:
            hand_shape = match.group(1)
            template_path = os.path.join(temp_dir, file)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # Load templates in color
            if template is None:
                print(f"Error: Could not load template image {template_path}")
                continue
            cropped_template = template[y:y+h, x:x+w]
            if hand_shape not in templates:
                templates[hand_shape] = []
            templates[hand_shape].append(cropped_template)
    return templates

def init_background(bg):
    return cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

def match_template(frame, templates, skin):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    scores = {}
    for hand_shape, template_imgs in templates.items():
        for template in template_imgs:
            if template.shape != gray_frame.shape:
                template = cv2.resize(template, (gray_frame.shape[1], gray_frame.shape[0]))
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            max_val = cv2.minMaxLoc(res)[1]
            if hand_shape not in scores or max_val > scores[hand_shape]:
                scores[hand_shape] = max_val
    if scores:
        best_match = max(scores, key=scores.get)
        best_score = scores[best_match]
        return best_match, best_score
    else:
        return None, 0
    
def main():
    video = cv2.VideoCapture(0)
    ret, frame = video.read()
    x, y, w, h = 50, 50, 300, 300
    bg_frame = init_background(frame[y:y+h, x:x+w])
    prev_frame = frame[y:y+h, x:x+w]
    templates = load_templates()
    #video writing
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file = 'hand_shapes_2.avi'
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #creating the output file objects
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))
    bg_mask_out = cv2.VideoWriter('bg_diff_mask.avi', fourcc, 20.0, (frame_width, frame_height))
    frame_diff_mask_out = cv2.VideoWriter('frame_diff_mask.avi', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = video.read()
        if not ret:
            break
        cropped_frame = frame[y:y+h, x:x+w]
        #detect the skin 
        skin = detect_skin(cropped_frame)
        bg_diff_mask = background_difference(bg_frame, cropped_frame)
        frame_diff_mask = frame_difference(prev_frame, cropped_frame)
        hand_shape, score = match_template(cropped_frame, templates, skin)
        contours, _ = cv2.findContours(bg_diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            circularity = calculate_circularity(largest_contour)
        else:
            circularity = 0
        if circularity > 0.45:
            hand_shape = "closed_fist"
        if score <= 0.25:
            hand_shape = "None"
            score = 0 
        #adding hand shape annotations and circularity information
        cv2.putText(frame, f'Hand Shape: {hand_shape}, Score: {score:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Circularity: {circularity:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        #displaying the 3 video feeds
        cv2.imshow("Live Video Feed", frame)
        cv2.imshow("Background Difference Mask", bg_diff_mask)
        cv2.imshow("Frame-Frame Difference Mask", frame_diff_mask)
        #writing the 3 feeds to video objects
        out.write(frame)
        bg_mask_out.write(bg_diff_mask)
        frame_diff_mask_out.write(frame_diff_mask)

        prev_frame = cropped_frame

        if cv2.waitKey(30) == 27:
            break

    video.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()