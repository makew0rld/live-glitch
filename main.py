import numpy as np
import cv2
import time
from PIL import ImageFont, ImageDraw, Image
import random
import importlib
import sys
from fps import FPS
import dlib
import os
import argparse
import imutils
import imutils.face_utils
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS
# Import my modified imutils.video package
MODULE_PATH = os.path.join("imutils_video", "__init__.py")
MODULE_NAME = "imutils.video"
spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
imutils_video = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = imutils_video 
spec.loader.exec_module(imutils_video)

# Use "." for same directory
RESOURCE_PATH = "resources"
SCREENSHOTS_PATH = "screenshots"
RECORDINGS_PATH = "recordings"

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--resize-width", default=500, type=int, help="The internal image width for processing.")
ap.add_argument("-d", "--display-width", type=int, help="The width of the image displayed on the screen.")
args = vars(ap.parse_args())
print(args)


print("[INFO] loading models...")
face_net_dnn = cv2.dnn.readNetFromCaffe(os.path.join(RESOURCE_PATH, "deploy.prototxt.txt"), os.path.join(RESOURCE_PATH, "res10_300x300_ssd_iter_140000.caffemodel"))
bg_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
facial_landmarker = dlib.shape_predictor(os.path.join(RESOURCE_PATH, "shape_predictor_68_face_landmarks.dat"))

# Initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = imutils_video.VideoStream().start()  # Use max camera resolution
time.sleep(2.0)  # Wait for the camera to start


# Read base frame to get dimensions
frame = vs.read()

# Set it to display at the width of the webcam
if args["display_width"] is None:
    args["display_width"] = frame.shape[1]
    print("Using display width of", args["display_width"])

# Set resizing width to improve performance and FPS
if args["display_width"] < args["resize_width"]:
    print("No internal resizing will be done, because the output value is smaller than the resizing value.")
    RESIZE_WIDTH = None
else:
    RESIZE_WIDTH = args["resize_width"]
    print("Using resize width of", RESIZE_WIDTH)

small_frame = imutils.resize(frame, width=RESIZE_WIDTH)
cam_h, cam_w = small_frame.shape[:2]  # Cam dimensions for normal smaller frame
# Resize up and get dimensions after resizing
# That's because sometimes after sizing it down and then up, the dims are slightly different
big_frame = imutils.resize(small_frame, width=args["display_width"])
MAX_CAM_H = big_frame.shape[0]  # Store for resizing later
MAX_CAM_W = big_frame.shape[1]


VHS_TEXT_PC = 0.0625
vhs_font = ImageFont.truetype(os.path.join(RESOURCE_PATH, "VCR_OSD_MONO_1.001.ttf"), int(cam_h * VHS_TEXT_PC))
UNI_FONT_EXTRA_PTS_PC = 1/24  # Font has to be a bit larger to match the size I want for characters like: ▶
uni_font = ImageFont.truetype(os.path.join(RESOURCE_PATH, "unifont-12.1.04.ttf"), int(cam_h * VHS_TEXT_PC) + int(cam_h * UNI_FONT_EXTRA_PTS_PC))


def find_faces_dnn(img, min_confidence=0.6):
    """Finds faces within the given image, and returns the confidences, and coords arrays.

    Uses opencv's DNN method.

    Returns: (confidences, coords)
    confidences:
        np.array([1, 0.96, 0.78, ...], dtype="float")
    coords array: 2D array
        np.array(
            [[startX, startY, endX, endY],
            [startX2, startY2, endX2, endY2],
            ...]
        dtype="uint16")

    Returns two empty numpy arrays if no faces are found.
    """

    # grab the frame dimensions and convert it to a blob
    (h, w) = img.shape[:2]
    # XXX: Parameters are magic numbers from the pyimagesearch.com tutorial
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    face_net_dnn.setInput(blob)
    detections = face_net_dnn.forward()

    confidences = []
    coords = []
    # Loop over each found face
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < min_confidence:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        box = box.astype("int")
        # Make sure it's not out of image bounds
        box[0] = sorted((0, box[0], w - 1))[1]
        box[1] = sorted((0, box[1], h - 1))[1]
        box[2] = sorted((0, box[2], w - 1))[1]
        box[3] = sorted((0, box[3], h - 1))[1]
        box = box.astype("uint16")  # No more negative numbers
        # Store them to return at the end
        confidences.append(confidence)
        coords.append(box)

    return (np.array(confidences, dtype="float"), np.array(coords, dtype="uint16"))


def find_facial_landmarks(face_boxes, img, dnn_used=True, index=None):
    """Returns an array of an array of facial landmark points, 68 per face.

    This uses a dlib trained model to identify different parts of the face.
    Each of the points in order:
        https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    More info:
        https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

    face_boxes: 2D numpy array of the following coords:
        np.array([startX, startY, endX, endY], dtype="uint16")
    index:
        An integer specifying which face to process. Optional.
    dnn_used:   Boolean indicating whether OpenCV dnn was used for face detection or not
                It's assumed the dlib HOG detector was used if set to False.

    Returns: A 2D array of points, but for each face. The result is a 3D array.
    Each array of points looks like this:
        np.array([[x1, y1], [x2, y2], ...], dtype="uint16")
    """

    face_boxes2 = face_boxes.copy()  # Don't modify global variables!

    if not index is None:
        face_boxes2 = face_boxes2[index]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    all_pts = []
    for i, box in enumerate(face_boxes2):
        if dnn_used:
            # Make the face box more square - dlib expects that for accurate detection

            # Alternate method I didn't use:
            #   https://stackoverflow.com/a/54383224/7361270
            # Current method:
            # This code changes some values for supposed "better landmark detection":
            #   https://github.com/keyurr2/facial-landmarks/blob/master/facial_landmarks.py#L88
            #   y1, x2 = int(y1 * 1.15), int(x2 * 1.05)
            # It does actually work well, for some reason
            box[1] = int(box[1] * 1.15)
            box[2] = int(box[2] * 1.05)

        # Convert face_boxes to dlib rectangles
        rect = dlib.rectangle(box[0], box[1], box[2], box[3])  
        shape = facial_landmarker(gray, rect)
        all_pts.append(imutils.face_utils.shape_to_np(shape, dtype="uint16"))
    return np.array(all_pts, dtype="uint16")


def find_blackout_bar(face_pts, face_box):
    """Returns a contour for a blackout bar that covers the two eyes on the given face.

    Input eyes format:
        [[x1, y1], [x2, y2], [x2, y3], [x4, y4]]  # For an angled rect
    
    Both formats should be numpy arrays.

    face_pts:   It should be a 2D numpy array of the 68 dlib points.
                https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    face_box:   The bounding rect for the face.
                Format: np.array([startX, startY, endX, endY])

    Returns: the four contour/box vertices for the blackout bar.
    Easily displayed with cv2.drawContours.
    """

    # Create a RotatedRect: ((centerX, centerY), (w, h), angle)
    # Find the center point - between each eye, on the nose
    # Point 28 could be used, but this is more accurate
    # Find the angle between the eyes for the rect as well

    # First get eyes and find the middle point of each by averaging
    # Source: https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
    left_pts = face_pts[lStart:lEnd]
    right_pts = face_pts[rStart:rEnd]    
    left_center = left_pts.mean(axis=0).astype("int")
    right_center = right_pts.mean(axis=0).astype("int")
    eyes_center = ((left_center[0] + right_center[0]) // 2,
                   (left_center[1] + right_center[1]) // 2)
    # Compute the angle between the eye centroids
    dY = right_center[1] - left_center[1]
    dX = right_center[0] - left_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # Decide on width and height, using the face_box
    face_w = face_box[2] - face_box[0]
    face_h = face_box[3] - face_box[1]
    width = face_w * 1.20  # 10% wider
    height = face_h * 0.25  # 25% of the face height
    # Create the rotated rectangle
    rrect = ((eyes_center[0], eyes_center[1]), (width, height), angle)
    return cv2.boxPoints(rrect).astype("int32")


def get_save_name():
    """Returns a string in the preferred save filename format."""

    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def screenshot(img):
    cv2.imwrite(SCREENSHOTS_PATH + "/" + get_save_name() + ".jpg", img)
    #cv2.putText(ui_frame, "Saved!", (cam_w//2, cam_h//2), cv2.FONT_HERSHEY_SIMPLEX, cam_h / 720, (0, 255, 0), 2)


def drag_face(face_box, img, line=None, angle=None, direction=None):
    """Modifies an image in place with the face dragged.
    
    face_box:   np.array([startX, startY, endX, endY], dtype="int")
    img:        BGR image that is larger than the face_box
    line:       Integer that represents the column or row of the face
                that should be dragged.
                TODO: What about other orientations?

    angle:      "v" or "h" for vertical or horizonatal
                The angle of the line that gets dragged.
    direction:  "down" or "up" - down also means left and up is right
                What direction the drag goes in.

    Some parameters default to None, meaning they will be randomly chosen.
    
    TODO: Support all angles.
    """

    # XXX: Shrink the box to prevent face edges being dragged?
    startX, startY, endX, endY = face_box
    h, w = img.shape[:2]

    if angle is None:
        angle = random.choice(["v", "h"])
    elif angle not in ["v", "h"]:
        raise ValueError('Angle must be either "v" or "h".')

    if direction is None:
        direction = random.choice(["up", "down"])
    elif direction not in ["up", "down"]:
        raise ValueError('direction must be either "up" or "down".')

    # Pick a random pixel column from the face
    # Then copy it over to a random direction
    
    if angle == "v":  # Vertical
        if line is None:
            column = random.randint(startX, endX)
        else:
            column = line
        
        if direction == "up":  # Left
            img[startY:endY, :column] = img[startY:endY, column:column+1]
        else:  # Right
            img[startY:endY, column:w-1] = img[startY:endY, column:column+1]
    
    else:  # Horizontal
        if line is None:
            row = random.randint(startY, endY)
        else:
            row = line
        
        if direction == "up":
            img[:row, startX:endX] = img[row:row+1, startX:endX]
        else:  # Down
            img[row:h-1, startX:endX] = img[row:row+1, startX:endX]


def draw_vhs_text(img, top="PLAY", bottom=None, top_margin_pc=1/16,
                  side_margin_pc=1/25):
    """Draws text in a VHS font and returns the modified image.
    
    top:    Text in the top left corner
    bottom: Text in the bottom left. Displays the time if it's 
            set to None.

    Source: https://stackoverflow.com/a/46558093/7361270
    """
    
    h, w = img.shape[:2]

    # Create the base image and convert it
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # Top

    # Button symbol
    if top == "PAUSE":
        symbol = "⏸"
    else:
        symbol = "▶"

    button_width, _ = uni_font.getsize(symbol)
    # The regular height, minus the extra height from the extra large font size
    # This way the play button is centered with the text in the other font
    y = int(h * top_margin_pc) - int(h * UNI_FONT_EXTRA_PTS_PC)//2
    draw.text((int(w * side_margin_pc), y), symbol, font=uni_font, fill=(255, 255, 255))
    draw.text((int(w * side_margin_pc) + button_width, int(h * top_margin_pc)), top, font=vhs_font, fill=(255, 255, 255))
    
    # Bottom
    if bottom is None:
        cur_time = time.localtime()
        month = time.strftime("%b", cur_time).upper() + "."
        _time = time.strftime("%p %I:%M", cur_time)
        date = month + " " + time.strftime("%d %Y", cur_time)
        bottom = _time + "\n" + date
    
    # Subtract the height of two lines, and then some
    y = int(h - (h * VHS_TEXT_PC) * 2 - h * top_margin_pc)
    x = int(w * side_margin_pc)
    draw.text((x, y), bottom, font=vhs_font, fill=(255, 255, 255))
    
    # Convert it back
    img = np.array(img_pil, dtype="uint8")
    return img


def strip_shift_face(face_box, img, orientation=None, n_strips=16, 
                     move_min_pc=0.1, move_max_pc=0.4, saved_moves=None):
    """Divides the face into vertical or horizontal strips, then shifts them around.

    face_box:       np.array([startX, startY, endX, endY], dtype="int")
    orientation:    "h" or "v", or None for random selection
    n_strips:       The number of strips the face is divided into. If
                    this is higher than the number of rows/columns of
                    the face, then each row/column is a strip.
    move_min_pc:    The minimum percentage of the face width/height that
                    a strip will move out by.
    move_min_pc:    The maximum percentage of the face width/height that
                    a strip will move out by.
    saved_moves:    An array of the shift values for each strip. This is
                    also calculated and returned by the function.
                    Passing the array here can allow control over where
                    each strip moves.
                    It must be the same length as n_strips.

    The actual movement of each strip is randomly calculated from those
    two percentages, inclusive.

    This is also known as "face glitching".

    Returns: saved_moves
    The image is modified in place.
    """

    if saved_moves is None or saved_moves == []:
        saved_moves = []
    else:
        assert len(saved_moves) == n_strips

    startX, startY, endX, endY = face_box
    face_h = endY - startY
    face_w = endX - startX
    img_h, img_w = img.shape[:2]

    if orientation is None:
        orientation = random.choice(["h", "v"])
    
    if orientation == "h":  # Horizontal
        strip_width = face_h // n_strips
    else:  # Vertical
        strip_width = face_w // n_strips

    if strip_width == 0:
        # There were more strips than pixels
        strip_width = 1
        # Set n_strips to the number of pixel rows/columns
        if orientation == "h":
            n_strips = face_h
        else:
            n_strips = face_w
    
    if orientation == "h":

        # Pre-calculate random shifts/moves for each strip
        if saved_moves == []:  # If there's no previously saved moves
            for i in range(n_strips):
                saved_moves.append(int(random.uniform(move_min_pc * face_w, move_max_pc * face_w)))
                # Make some of them go to the opposite side - negative shift
                saved_moves[i] *= random.choice([-1, 1])

        for i in range(n_strips):
            strip_startY = startY + i * strip_width
            strip_endY = startY + i * strip_width + strip_width
            shift = saved_moves[i]
            
            strip_startX = startX + shift
            strip_endX = endX + shift
            # Check if this will go outside the image boundaries and correct
            shrink_l, shrink_r = 0, 0  # How much the face width should shrink by, if the strip needs to be shrunk bc it goes outside boundaries
            if strip_startX < 0:
                shrink_l = abs(strip_startX)
                strip_startX = 0
            if strip_endX > img_w:
                shrink_r = strip_endX - img_w
                strip_endX = img_w

            # Replace the strip X-area with face X-area for that Y section
            img[strip_startY:strip_endY, strip_startX:strip_endX] = img[strip_startY:strip_endY, startX + shrink_l:endX - shrink_r]

    else:  # Vertical
        # Look at horizontal version above for comments

        if saved_moves == []:
            for i in range(n_strips):
                saved_moves.append(int(random.uniform(move_min_pc * face_w, move_max_pc * face_w)))
                saved_moves[i] *= random.choice([-1, 1])

        for i in range(n_strips):
            strip_startX = startX + i * strip_width
            strip_endX = startX + i * strip_width + strip_width
            shift = saved_moves[i]

            strip_startY = startY + shift
            strip_endY = endY + shift

            shrink_t, shrink_b = 0, 0  # Top and bottom
            if strip_startY < 0:
                shrink_t = abs(strip_startY)
                strip_startY = 0
            if strip_endY > img_h:
                shrink_b = strip_endY - img_h
                strip_endY = img_h
            
            # Replace the strip Y-area with face Y-area for that X section
            img[strip_startY:strip_endY, strip_startX:strip_endX] = img[startY + shrink_t:endY - shrink_b, strip_startX:strip_endX]

    return saved_moves


def add_scanlines(img, width_pc=1/150, darken=15):
    """Adds VHS style scanlines to an image and returns nothing.

    width_pc:   The percentage of the image that each scanline
                should take up, width-wise (y-axis). From 0 to 1.
                A value like 1/30 will mean there will be 30
                scanlines in the returned image
    """

    h, w = img.shape[:2]
    # Number of scanlines
    n_lines = int(h / (width_pc * 100))
    width_px = int(h * width_pc)

    for i in range(n_lines):
        # Each scanline area just has `darken` subtracted from it, making sure it doesn't go below 0
        dark_line = np.maximum(img[i*width_px*2 : i*width_px*2 + width_px, :].astype("int16") - darken, 0).astype("uint8")
        img[i*width_px*2 : i*width_px*2 + width_px, :] = dark_line


def add_vhs_color_distortion(img, spacing=5):
    """Adds blue and red image copies, shifted to the left and right.

    spacing:    How much the copies are shifted, in pixels.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Create blue and red parts using HSV - blue is NOT full RGB/BGR blue
    sat = np.full(gray.shape, 100, dtype="uint8")
    blue_hue = np.full(gray.shape, 180//2, dtype="uint8")  # opencv uses 0-179 for hue
    red_hue = np.full(gray.shape, 0, dtype="uint8")
    # The value part of HSV is from the grayscale image
    blue = cv2.merge(np.array([blue_hue, sat, gray], dtype="uint8"))
    red = cv2.merge(np.array([red_hue, sat, gray], dtype="uint8"))
    # Convert
    blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)
    red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)
    # Shift them over
    blue = imutils.translate(blue, spacing, 0)  # Left
    red = imutils.translate(red, -spacing, 0)  # Right
    # Add them on top, using "lighten only" - only higher value pixels show up
    np.copyto(img, blue, where=(blue > img))
    np.copyto(img, red, where=(red > img))
    # Increase saturation and value
    result_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(result_hsv)
    s *= 1.5  # Add saturation
    s = np.clip(s, 0, 255)
    v += 20  # Brighten image
    v = np.clip(v, 0, 255)
    result_hsv = cv2.merge([h, s, v])
    result = cv2.cvtColor(result_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
    return result


def resize_points(pts, orig_hw, new_hw):
    """Moves points on an image so they are in the same relative place in a resized image.

    pts:        np.array([[x1, y1], [x2, y2], ...])
                Points on the original image.
    
    orig_hw:    (height, width) of the original image
    new_hw:     (height, width) of the new image

    Returns an array of the same length and type as `pts`.
    The recommended dtype is `uint16`.
    """

    pts2 = pts.copy().astype("float")

    h_rat = new_hw[0] / orig_hw[0]
    w_rat = new_hw[1] / orig_hw[1]

    pts2[:, 0] *= w_rat
    pts2[:, 1] *= h_rat

    return pts2.astype("uint16")


def resize_boxes(boxes, orig_hw, new_hw):
    """Resizes bounding rectanges/boxes so they are in the same relative place in a resized image.

    boxes: 2D numpy array of the following coords:
        np.array([startX, startY, endX, endY], dtype="uint16")
    
    orig_hw:    (height, width) of the original image
    new_hw:     (height, width) of the new image

    Returns an array of the same length and type as `boxes`.
    The recommended dtype is `uint16`.
    """

    boxes2 = boxes.copy().astype("float")  # Don't modify them in place!

    # Use ratios
    h_rat = new_hw[0] / orig_hw[0]
    w_rat = new_hw[1] / orig_hw[1]

    boxes2[:, 0] *= w_rat
    boxes2[:, 2] *= w_rat
    boxes2[:, 1] *= h_rat
    boxes2[:, 3] *= h_rat

    return boxes2.astype("uint16")


def tears(face_pts, img, length=None, n_tears=8, variance=0.5, saved_moves=None):
    """Adds glitch tears coming from the eyes.

    face_pts:   The 68 dlib facial landmarking points.
    length:     A decimal from 0 to 1, indicating how far the
                tears should be as a percentage. A value of 1
                means the tears will always reach the bottom,
                while a value of 0.5 means the tears will stretch
                to half the image height.
    n_tears:    The number of tears/strips that will extend from
                the eye.
    variance:   A decimal from 0 to 1 indicating how close to the
                `length` the tears will go. 1 means they will all
                have random lengths, with `length` as the max, and
                0 means they will all be exactly `length`.
    saved_moves: This function returns the value you can use here.
                Passing this value will mean the eyes are still tracked,
                and the tears are still drawn, but the length of each
                tear is the same.

    Returns: saved_moves - for use in the saved_moves args.

    The image is modified in place.
    """

    if length == 0:
        return

    max_tear_h = int(length * img.shape[0])
    if max_tear_h == 0:
        max_tear_h = 1

    if saved_moves is None:
        moves = [[], []]  # Left and right
    else:
        moves = saved_moves

    # Find the Y-axis line for each eye, by finding the middle
    # That's the line that the tears come down from
    left_y = (face_pts[36, 1] + face_pts[39, 1]) // 2
    right_y = (face_pts[42, 1] + face_pts[45, 1]) // 2
    # X-axis bounds are found using the same outer eye points
    left_x1 = face_pts[36, 0]
    left_x2 = face_pts[39, 0]
    left_w = left_x2 - left_x1
    right_x1 = face_pts[42, 0]
    right_x2 = face_pts[45, 0]
    right_w = right_x2 - right_x1
    
    # Calculate the width of each tear in pixels
    left_tear_w = left_w // n_tears
    if left_tear_w == 0:
        left_tear_w = 1
    right_tear_w = right_w // n_tears
    if right_tear_w == 0:
        right_tear_w = 1

    for col in range(n_tears):
        # Left eye tears
        # The length of each drip is randomized, but stays close to max
        if saved_moves is None:
            moves[0].append(random.randint(int((1-variance) * max_tear_h), max_tear_h))
        
        drop = sorted((1, left_y + moves[0][col], img.shape[0]))[1]  # Keep within image bounds
        tear_startX = left_x1 + left_tear_w*col
        tear_endX = left_x1 + left_tear_w*col + left_tear_w
        # Drag downwards
        img[left_y:drop, tear_startX:tear_endX] = img[left_y, tear_startX:tear_endX]
        
        # Right eye tears - see above for comments
        if saved_moves is None:
            moves[1].append(random.randint(int((1-variance) * max_tear_h), max_tear_h))
        
        drop = sorted((1, left_y + moves[1][col], img.shape[0]))[1]
        tear_startX = right_x1 + right_tear_w*col
        tear_endX = right_x1 + right_tear_w*col + right_tear_w
        img[right_y:drop, tear_startX:tear_endX] = img[right_y, tear_startX:tear_endX]

    return moves


def gaussian_noise(img, variance):
    # Slow.

    # Source: https://stackoverflow.com/a/30609854/7361270
    row, col, ch = img.shape
    mean = 0
    sigma = variance ** 1
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX)
    return noisy.astype("uint8")


# Store all the options in a dict instead of having many many global variables
options = {
    "blackout_bar": {
        "state": False,
        "prev_bar": None,       # Place to store the bar contour from previous frames
        "prev_bar_time": 10000, # The time when the prev_bar was last stored
        "interval": 0.5         # The max amount of time (secs) that can pass before prev_bar can't be used anymore
    },
    "face_drag": {
        "state": False, 
        "angle": "v",
        "direction": "up",  # Left
        "rel_line": None  # Relative to the face startX or startY - divided by width/height
    },
    "vhs_text": {
        "state": True
    },
    "scanlines": {
        "state": True
    },
    "vhs_color_distortion": {
        "state": False
    },
    "face_glitch": {
        "state": False,
        "orientation": "h",
        "saved_moves": None,
        "interval": 0.2,        # How often the glitch bars randomize, in seconds
        "start_time": -1        # The time of the last randomization, using time.time()
    },
    "show_debug": {
        "state": False
    },
    "tears": {
        "state": False,
        "saved_moves": None,
        "move_interval": 0,     # How often the tears randomize, in seconds
        "move_start": -1,       # The time of the last randomization, using time.time()
        "grow_interval": 0,     # How often the length grows, in seconds
        "grow_amount": 0.05,    # How much is added to the length
        "length": 0.1,          # Starting length
        "grow_start": -1        # The time of the last growing, using time.time()
    }
}

# Arrow keys, as recorded on Arch Linux XFCE with OpenCV 4.2.0
KEY_UP      = 82
KEY_DOWN    = 84
KEY_LEFT    = 81
KEY_RIGHT   = 83

# Percentage that the face drag line shifts when an arrow key is pressed
FACE_DRAG_PC = 0.02


# Start recording FPS
fps = FPS().start()
# Empty video recording object for when the user wants
record = None
RECORD_FPS = 20.0  # The fps of the output video file

while True:
    # Get frame and create other layers
    frame = vs.read()
    frame = cv2.flip(frame, 1)  # Flip around y-axis to resemble a mirror
    frame = imutils.resize(frame, width=RESIZE_WIDTH)
    draw_frame = frame.copy()
    debug_frame = np.zeros(frame.shape, dtype="uint8")  # BGR mask of debug info
    ui_frame = np.zeros(frame.shape, dtype="uint8")  # BGR mask of UI elements

    # Display FPS on debug screen
    cv2.putText(debug_frame, f"FPS: {fps.fps():.1f}", (cam_w - cam_w//8, cam_h//16), cv2.FONT_HERSHEY_SIMPLEX, cam_h / 720, (0, 255, 0), 1)

    # Options that require face detection
    # TODO: Fix the long if statement
    if options["blackout_bar"]["state"] or options["face_drag"]["state"] or options["face_glitch"]["state"] or options["tears"]["state"]:
        _, face_boxes = find_faces_dnn(frame)

        if len(face_boxes) > 0:
            # Find the facial landmarks, if the options require it
            if options["tears"]["state"] or options["blackout_bar"]["state"]:

                # ------- Display the facial landmarking debug info ---------
                small_frame = imutils.resize(frame, width=500)  # Facial landmarking is intensive
                # Align face_boxes to small frame
                small_face_boxes = resize_boxes(face_boxes, frame.shape[:2], small_frame.shape[:2])
                all_faces_pts = find_facial_landmarks(small_face_boxes, small_frame)
                for i in range(all_faces_pts.shape[0]):  # pylint: disable=unsubscriptable-object
                    # Realign small landmarks to large frame
                    all_faces_pts[i] = resize_points(all_faces_pts[i], small_frame.shape[:2], frame.shape[:2])
                    for (x, y) in all_faces_pts[i]:
                        cv2.circle(debug_frame, (x, y), 1, (0, 255, 0), -1)

            for i, face_box in enumerate(face_boxes):
                face_w = face_box[2] - face_box[0]
                face_h = face_box[3] - face_box[1]
                cv2.rectangle(debug_frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (255, 0, 0), 1)

                if options["face_drag"]["state"]:
                    if options["face_drag"]["rel_line"] is None:
                        options["face_drag"]["rel_line"] = 0.5  # Middle
                    # Find the actual line, relative to the image
                    if options["face_drag"]["angle"] == "v":
                        line = int(options["face_drag"]["rel_line"] * face_w) + face_box[0]
                    else:  # Horizontal
                        line = int(options["face_drag"]["rel_line"] * face_h) + face_box[1]

                    drag_face(face_box, draw_frame, line, options["face_drag"]["angle"], options["face_drag"]["direction"])

                elif options["face_glitch"]["state"]:  # Can't have glitch and drag, so elif is used
                    if time.time() - options["face_glitch"]["start_time"] >= options["face_glitch"]["interval"]:
                        # Enough time has passed, reset the saved moves and timer
                        options["face_glitch"]["saved_moves"] = None
                        options["face_glitch"]["start_time"] = time.time()

                    saved_moves = strip_shift_face(face_box, draw_frame, 
                                                   options["face_glitch"]["orientation"],
                                                   saved_moves=options["face_glitch"]["saved_moves"])
                    options["face_glitch"]["saved_moves"] = saved_moves

                elif options["tears"]["state"]:
                    # Check if enough time has passed, reset the saved moves and timer
                    if time.time() - options["tears"]["move_start"] >= options["tears"]["move_interval"]:
                        options["tears"]["saved_moves"] = None
                        options["tears"]["move_start"] = time.time()
                    
                    # Check if it's time for the tears to grow
                    if time.time() - options["tears"]["grow_start"] >= options["tears"]["grow_interval"]:
                        if options["tears"]["length"] >= 1:
                            options["tears"]["length"] = 0.1
                        else:
                            options["tears"]["length"] += options["tears"]["grow_amount"]
                        options["tears"]["grow_start"] = time.time()

                    saved_moves = tears(all_faces_pts[i], draw_frame, options["tears"]["length"],
                                        saved_moves=options["tears"]["saved_moves"])
                    options["tears"]["saved_moves"] = saved_moves

                # Do blackout bar on top of other glitches
                if options["blackout_bar"]["state"]:
                    # Try using facial landmarking first, fall back to haar detection otherwise
                    bar = find_blackout_bar(all_faces_pts[i], face_box)
                    
                    # The bar never doesn't appear, it's just wrong sometimes. Is there a way to fix this?

                    # if bar is None and (not options["blackout_bar"]["prev_bar"] is None) and \
                    #     time.time() - options["blackout_bar"]["prev_bar_time"] <= options["blackout_bar"]["interval"]:
                    #     # No bar was found, but an old one can be used
                    #     bar = options["blackout_bar"]["prev_bar"]
                    # if not bar is None:

                    cv2.drawContours(draw_frame, [bar], 0, (0, 0, 0), -1)
                    # Store it for later
                    options["blackout_bar"]["prev_bar"] = bar
                    options["blackout_bar"]["prev_bar_time"] = time.time()

    # Options that are drawn on top of everything
    #if options["vhs_color_distortion"]["state"]:  # TODO: Re-enable this after optimization
    #    draw_frame = add_vhs_color_distortion(draw_frame)
    if options["scanlines"]["state"]:
        add_scanlines(draw_frame)
    if options["vhs_text"]["state"]:
        draw_frame = draw_vhs_text(draw_frame)

    # Check keys before displaying UI and debug info
    # TODO: Fix this UX - onscreen buttons?
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("s"):
        screenshot(draw_frame)
    # Toggles
    elif key == ord("b"):
        options["blackout_bar"]["state"] = not options["blackout_bar"]["state"]
    elif key == ord("f"):
        options["face_drag"]["state"] = not options["face_drag"]["state"]
        options["face_glitch"]["state"] = False  # Can't have both
        options["tears"]["state"] = False
    elif key == ord("v"):  # For all VHS effects
        options["vhs_text"]["state"] = not options["vhs_text"]["state"]
        options["scanlines"]["state"] = not options["scanlines"]["state"]
        options["vhs_color_distortion"]["state"] = not options["vhs_color_distortion"]["state"]
    elif key == ord("g"):
        options["face_glitch"]["state"] = not options["face_glitch"]["state"]
        options["face_drag"]["state"] = False  # Can't have both
        options["tears"]["state"] = False
    elif key == ord("d"):
        options["show_debug"]["state"] = not options["show_debug"]["state"]
    elif key == ord("t"):
        options["tears"]["state"] = not options["tears"]["state"]
        options["face_glitch"]["state"] = False
        options["face_drag"]["state"] = False
    elif key == ord("r"):
        if record is None:
            # Create recorder - don't save anything yet
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # XXX: Linux only?
            record = cv2.VideoWriter(os.path.join(RECORDINGS_PATH, get_save_name() + ".mp4"), fourcc,
                                     RECORD_FPS, (cam_w, cam_h))
        else:
            # Stop the current recording
            record.release()
            record = None

    # Sub-key checks
    elif options["face_glitch"]["state"]:
        if key == ord("/"):
            if options["face_glitch"]["orientation"] == "v":
                options["face_glitch"]["orientation"] = "h"
            else:
                options["face_glitch"]["orientation"] = "v"
        elif key == KEY_UP and options["face_glitch"]["interval"] >= 0.1:
                options["face_glitch"]["interval"] -= 0.1
                print(options["face_glitch"]["interval"])
        elif key == KEY_DOWN and options["face_glitch"]["interval"] <= 10 - 0.1:
            options["face_glitch"]["interval"] += 0.1
            print(options["face_glitch"]["interval"])

    # Only check these keys if face dragging is on
    elif options["face_drag"]["state"]:
        if key == KEY_UP:
            if (not options["face_drag"]["rel_line"] is None) and options["face_drag"]["rel_line"] >= FACE_DRAG_PC:
                options["face_drag"]["rel_line"] -= FACE_DRAG_PC
        elif key == KEY_DOWN:
            if not options["face_drag"]["rel_line"] is None and options["face_drag"]["rel_line"] <= 1 - FACE_DRAG_PC:
                options["face_drag"]["rel_line"] += FACE_DRAG_PC
        elif key == KEY_LEFT:
            options["face_drag"]["direction"] = "up"
        elif key == KEY_RIGHT:
            options["face_drag"]["direction"] = "down"
        elif key == ord("/"):
            if options["face_drag"]["angle"] == "v":
                options["face_drag"]["angle"] = "h"
            else:
                options["face_drag"]["angle"] = "v"


    draw_frame = gaussian_noise(draw_frame, 8)

    # Optionally show debug info on top of the frame
    if options["show_debug"]["state"]:
        debug_frame_mask = debug_frame.max(axis=2, keepdims=True) > 0  # pylint: disable=unexpected-keyword-arg
        np.copyto(draw_frame, debug_frame, where=debug_frame_mask)

    # Calculate and show UI elements
    if not record is None:
        # Save current frame, accounting for slow FPS
        if fps.fps() >= RECORD_FPS or fps.fps() == 0:
            record.write(draw_frame)
        else:
            # Write extra frames to make it appear the same in the video as in the app
            for i in range(round(RECORD_FPS / fps.fps())):
                record.write(draw_frame)
        # Display red recording dot
        cv2.circle(ui_frame, (cam_w - 10, 10), 5, (0, 0, 255), -1)
    

    draw_frame = imutils.resize(draw_frame, width=MAX_CAM_W)

    ui_frame = imutils.resize(ui_frame, width=MAX_CAM_W)
    ui_frame_mask = ui_frame.max(axis=2, keepdims=True) > 0  # pylint: disable=unexpected-keyword-arg
    np.copyto(draw_frame, ui_frame, where=ui_frame_mask)

    cv2.imshow("Output", draw_frame)
    fps.update()

# Close up resources and quit
if not record is None:
    record.release()
cv2.destroyAllWindows()
vs.stop()
