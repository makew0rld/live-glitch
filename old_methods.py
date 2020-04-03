import numpy as np
import cv2


eye_cascade = cv2.CascadeClassifier(RESOURCE_PATH + "/haarcascade_eye_tree_eyeglasses.xml")


def find_eyes_haar(face_box, img):
    """Returns box coords for the two eyes of the face, given the box coords of the face and the base image.
    
    Finds eyes using haar cascades

    Returns: np.array([first_eye_box, ...], dtype="uint16")
    boxes:
        np.array([startY, endY, startX, endX], dtype="uint16")

    face_box:
        np.array([startX, startY, endX, endY], dtype="uint16")

    TODO: Remove completely? It's never really used as a fallback.
    """

    startX, startY, endX, endY = face_box
    h = endY - startY
    # Only consider the top bit of the face, crop out mouth and nose to reduce false positives
    face_img = img[startY:endY - int(h * 0.5), startX:endX]
    #cv2.imshow("Face crop", face_img)

    gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face_img, 1.05, 2)  # XXX: Tune these values - OG: (1.2, 6)
    eye_coords = []
    for (ex, ey, ew, eh) in eyes:  # Change to eyes[:2] to only capture two eyes?
        # Store the coords, relative to the total image
        eye_coords.append([startY + ey, startY + ey + eh, startX + ex, startX + ex + ew])
    
    # Find rects that are inside another one and remove them - sometimes this happens
    for eye in eye_coords:
        # Check each eye against all the other eyes
        for other_eye in eye_coords:
            if eye[0] > other_eye[0] and eye[1] < other_eye[1] and eye[2] > other_eye[2] and eye[3] < other_eye[3]:
                eye_coords.remove(eye)
                break

    return np.array(eye_coords[:2], dtype="uint16")


def gimp_overlay(lower, upper, opacity=0.5):
    """Overlays one image over the other, as defined by GIMP.

    Definition: https://docs.gimp.org/2.10/en/gimp-concepts-layer-modes.html#layer-mode-overlay
    
    opacity: From 0 to 1, determines the opacity of the upper image.
    Returns a single modified image.

    Unused.
    """

    assert upper.shape == lower.shape

    lower = lower.astype("float32")
    # Calculate opacity
    # Fill a layer with perfect gray, the only color that doesn't affect overlay
    opacity_layer = np.full(upper.shape, 50, dtype="float32") * (1 - opacity)
    upper = upper.astype("float32") * opacity
    upper = upper + opacity_layer

    e = (lower / 255) * (lower + ((2*upper)/255) * (255 - lower))
    return e.astype("uint8")


def create_eye_boxes(face_pts):
    """Returns to bounding rectangles, given 68 face points.

    The 68 points are a dlib standard.
        np.array([[x1, y1], [x2, y2], ...])
    
    Returns a tuple a two arrays of four points/vertices: left eye, then right eye.
    Each array format:
        np.array([[x1, y1], [x2, y2], [x2, y3], [x4, y4]], dtype="int")

    Integer types are returned for ease of use with cv2.drawContours.
    """

    # Point diagram:
    #   https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg
    # 6 points per eye
    left_pts = np.array(face_pts[36:42], dtype="int32")  # Can't be unsigned int when passed to minAreaRect
    right_pts = np.array(face_pts[42:48], dtype="int32")

    assert left_pts.shape == right_pts.shape
    assert left_pts.shape == (6, 2)

    left_rect = cv2.minAreaRect(left_pts)
    left_box = cv2.boxPoints(left_rect).astype("int32")
    right_rect = cv2.minAreaRect(right_pts)
    right_box = cv2.boxPoints(right_rect).astype("int32")

    return left_box, right_box


def sharpen(img, amount):
    # Unused
    # TODO: Try using https://stackoverflow.com/a/33971525/7361270 instead
    # Find the best sharpening method

    #blur = cv2.GaussianBlur(img, (0, 0), 3)
    #img = cv2.addWeighted(img, 1 + amount, blur, -amount, 0)
    #return img

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(img, -amount, kernel)


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask.
    
    Source: https://stackoverflow.com/a/55590133/7361270

    Not currently used. TODO: Optimize, make faster?
    """

    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened
