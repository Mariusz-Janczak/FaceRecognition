import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import cv2
import tempfile
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from PIL import Image, ImageDraw, ImageFont

FACE_BORDER = 1
TEXT_SIZE = 14
TEXT_COLOR = "black"
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", TEXT_SIZE)
except IOError:
    FONT = ImageFont.load_default()
    ascent, descent = FONT.getmetrics()
    TEXT_SIZE = ascent + descent


def resize_with_aspect_ratio(image, max_size: int = 1024):
    """
    Resize an image while maintaining its aspect ratio so that its largest dimension does not exceed max_size.

    Parameters:
        image (PIL.Image.Image): The input image to resize.
        max_size (int): The maximum allowed size for the largest dimension (default: 1024).

    Returns:
        tuple: A tuple containing the resized PIL Image and the scaling factor used.
    """
    scale = max_size / max(image.height, image.width)
    return image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS
    ), scale


def detect_faces(image_path: str, detector_backend: str = "retinaface", min_confidence: float = 0.5):
    """
    Detect faces in an image using the specified detector backend.

    Parameters:
        image_path (str): Path to the image file.
        detector_backend (str): The face detector backend to use (default: "retinaface").
        min_confidence (float): Minimum confidence threshold for detected faces (default: 0.5).

    Returns:
        list: A list of detected face dictionaries with confidence above the threshold.
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=detector_backend,
            enforce_detection=False
        )
        print(f"Detected {len(faces)} face(s) using '{detector_backend}' detector.")
        filtered_faces = [
            face for face in faces
            if face.get("confidence", 1.0) >= min_confidence
        ]
        if not faces:
            print("No faces detected.")
        elif len(faces) > len(filtered_faces):
            print(f"Filtered out {len(faces) - len(filtered_faces)} face(s) below confidence threshold of {min_confidence}.")
        return filtered_faces
    except Exception as e:
        print(f"Error during face detection: {e}")
        return []


def get_embeddings(model, faces: list, target_size):
    """
    Extract embeddings for a list of detected faces using the specified model and target size.

    Parameters:
        model: The face recognition model to use for embedding extraction.
        faces (list): A list of detected face dictionaries, each containing a "face" key with the face image.
        target_size: The target size for face images when extracting embeddings.

    Returns:
        np.ndarray: A 2D array of shape (num_faces, embedding_dim) containing the normalized embeddings for each
        detected face. If no faces are provided, returns an empty array with shape (0, embedding_dim).
    """
    if not faces:
        return np.empty((0, model.output_shape[-1]), dtype=np.float32)

    height, width = target_size[:2]
    batch = np.array(
        [cv2.resize(f["face"], (width, height)) for f in faces],
        dtype=np.float32
    )

    reps = model.forward(batch)
    if hasattr(reps, "detach"):
        reps = reps.detach().cpu().numpy()
    else:
        reps = np.array(reps, dtype=np.float32)

    reps = reps.reshape(reps.shape[0], -1)
    norms = np.linalg.norm(reps, axis=1, keepdims=True)
    reps /= np.clip(norms, 1e-10, None)

    return reps


def one_to_one_matching(faces1: list, faces2: list, model, target_size, metric: str, threshold: float):
    """
    Perform one-to-one matching of faces between two lists based on the specified distance metric and threshold.
    
    Parameters:
        faces1 (list): List of detected faces from the first image.
        faces2 (list): List of detected faces from the second image.
        model: The face recognition model to use for embedding extraction.
        target_size: The target size for face images when extracting embeddings.
        metric (str): The distance metric to use for comparison ("cosine", "euclidean", or "euclidean_l2").
        threshold (float): The distance threshold for considering a match.
    Returns:
        list: A list of tuples (i, j, distance) where i is the index of the face in faces1, j is the index of
        the face in faces2, and distance is the computed distance between their embeddings. Only matches with
        distance below the threshold are included.
    """
    if not faces1 or not faces2:
        print("No faces detected in one or both images.")
        return []

    print("Build embeddings...")
    embeddings1 = get_embeddings(model, faces1, target_size)
    embeddings2 = get_embeddings(model, faces2, target_size)

    if embeddings1.size == 0 or embeddings2.size == 0:
        print("No embeddings were produced.")
        return []

    print("Performing faces comparison...")
    if metric == "cosine":
        distance_matrix = 1.0 - np.dot(embeddings1, embeddings2.T)
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
    elif metric in ["euclidean", "euclidean_l2"]:
        if metric == "euclidean_l2":
            e1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
            e2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
        else:
            e1, e2 = embeddings1, embeddings2
        distance_matrix = cdist(e1, e2, metric="euclidean")
    else:
        raise ValueError(f"Unknown distance metric: {metric}")

    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matches = [(i, j, float(distance_matrix[i, j])) for i, j in zip(row_ind, col_ind) if distance_matrix[i, j] < threshold]
    return matches


def draw_id_with_background(draw, x: int, y: int, w: int, h: int, text: str, font, text_color: str = "black",
                            bg_color: str = "white", border: int = FACE_BORDER, padding: int = 2, vmargin: int = 2,
                            image_height: int | None = None):
    """
    Draws a text label with a background rectangle above or below a detected face bounding box.

    Parameters:
        draw: An ImageDraw.Draw object to draw on.
        x (int): The x-coordinate of the top-left corner of the face bounding box.
        y (int): The y-coordinate of the top-left corner of the face bounding box.
        w (int): The width of the face bounding box.
        h (int): The height of the face bounding box.
        text (str): The text label to draw (e.g., "ID 0").
        font: The font to use for the text.
        text_color (str): The color of the text (default: "black").
        bg_color (str): The background color for the text rectangle (default: "white").
        border (int): The width of the border around the face bounding box (default: FACE_BORDER).
        padding (int): The padding around the text inside the background rectangle (default: 2).
        vmargin (int): The vertical margin between the face bounding box and the text rectangle (default: 2).
        image_height (int | None): The height of the image to ensure the text rectangle does not go out of bounds (default: None, no constraint).

    Returns:
        None. The function modifies the provided ImageDraw object in place to add the text label with background.
    """
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    bbox_x0, bbox_y0 = bbox[0], bbox[1]

    desired_top = y - text_height - border - vmargin

    left = x + (w - text_width) / 2
    draw_x = int(round(left - bbox_x0))
    draw_y = int(round(desired_top - bbox_y0))

    if draw_y + bbox_y0 < 0:
        desired_top = y + h + border + vmargin
        draw_y = int(round(desired_top - bbox_y0))

    rect_left = int(round(draw_x + bbox_x0 - padding))
    rect_top = int(round(draw_y + bbox_y0 - padding))
    rect_right = int(round(draw_x + bbox[2] + padding))
    rect_bottom = int(round(draw_y + bbox[3] + padding))

    if image_height is not None:
        rect_top = max(0, rect_top)
        rect_bottom = min(image_height, rect_bottom)

    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=bg_color)
    draw.text((draw_x, draw_y), text, fill=text_color, font=font)


def extract_and_compare_faces(image1_path: str, image2_path: str, model_name: str = "ArcFace", detector: str = "retinaface", threshold: float | None = None):
    """
    Detects faces in two images, extracts embeddings using a specified face recognition model,
    compares the faces, and visualizes the results.

    Parameters:
        image1_path (str): Path to the first image file.
        image2_path (str): Path to the second image file.
        model_name (str): Name of the face recognition model to use (default: "ArcFace").
        detector (str): Name of the face detector backend to use (default: "retinaface").
        threshold (float, optional): Custom distance threshold for matching faces. If None, uses the model's default.

    Returns:
        None. Displays annotated images and prints matching results to the console.
    """
    if not os.path.isfile(image1_path) or not os.path.isfile(image2_path):
        print("Error: both image files are required.")
        return

    print("Loading images...")
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp1:
        img1.save(tmp1.name)
        tmp1_path = tmp1.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp2:
        img2.save(tmp2.name)
        tmp2_path = tmp2.name

    print(f"Detecting faces using '{detector}' detector...")
    faces1 = detect_faces(tmp1_path, detector)
    faces2 = detect_faces(tmp2_path, detector)
    
    os.unlink(tmp1_path)
    os.unlink(tmp2_path)

    print(f"Recognizing faces using '{model_name}' model...")
    model: FacialRecognition = DeepFace.build_model(task="facial_recognition", model_name=model_name)
    target_size = model.input_shape
    print(f"Faces target size: {target_size}")

    if model_name in ["ArcFace", "SFace", "Buffalo_L"]:
        metric = "cosine"
    elif model_name in ["Facenet", "Facenet512", "VGG-Face", "Dlib", "DeepFace"]:
        metric = "euclidean_l2"
    else:
        metric = "euclidean"

    default_threshold = verification.find_threshold(model_name=model_name, distance_metric=metric)
    if threshold is None:
        threshold = default_threshold

    print(f"Model metric: {metric}, Threshold: {threshold:.3f}")

    matches_found = one_to_one_matching(faces1, faces2, model, target_size, metric, threshold)

    img1_scaled, scale1 = resize_with_aspect_ratio(img1)
    img2_scaled, scale2 = resize_with_aspect_ratio(img2)

    draw1 = ImageDraw.Draw(img1_scaled)
    draw2 = ImageDraw.Draw(img2_scaled)

    for i, face in enumerate(faces1):
        x, y, w, h = [int(face["facial_area"][k] * scale1) for k in ("x", "y", "w", "h")]
        draw1.rectangle([x, y, x + w, y + h], outline="red", width=FACE_BORDER)

    for j, face in enumerate(faces2):
        x, y, w, h = [int(face["facial_area"][k] * scale2) for k in ("x", "y", "w", "h")]
        draw2.rectangle([x, y, x + w, y + h], outline="red", width=FACE_BORDER)

    for i, j, distance in matches_found:
        color = "green" if distance < min(threshold, default_threshold) else "yellow"
        x1, y1, w1, h1 = [int(faces1[i]["facial_area"][k] * scale1) for k in ("x", "y", "w", "h")]
        draw1.rectangle([x1, y1, x1 + w1, y1 + h1], outline=color, width=FACE_BORDER)
        text = f"ID {i}"
        draw_id_with_background(draw1, x1, y1, w1, h1, text, FONT, text_color=TEXT_COLOR, bg_color="white",
                                border=FACE_BORDER, padding=2, vmargin=2, image_height=img1_scaled.height)
        x2, y2, w2, h2 = [int(faces2[j]["facial_area"][k] * scale2) for k in ("x", "y", "w", "h")]
        draw2.rectangle([x2, y2, x2 + w2, y2 + h2], outline=color, width=FACE_BORDER)
        text = f"ID {j}"
        draw_id_with_background(draw2, x2, y2, w2, h2, text, FONT, text_color=TEXT_COLOR, bg_color="white",
                                border=FACE_BORDER, padding=2, vmargin=2, image_height=img2_scaled.height)

    img1_scaled.show(title=image1_path)
    img2_scaled.show(title=image2_path)

    if matches_found:
        print(f"Found {len(matches_found)} match(es):")
        reset = "\033[0m"
        for i, j, distance in matches_found:
            color = "\033[92m" if distance < min(threshold, default_threshold) else "\033[93m"
            print(f"{color}Face ID {i} of {image1_path} matches face ID {j} of {image2_path} (distance: {distance:.3f}){reset}")
    else:
        print("No matching faces found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFace comparison of faces found on input images.")
    parser.add_argument("image1", help="Path to the 1st image")
    parser.add_argument("image2", help="Path to the 2nd image")
    parser.add_argument("--model", default="ArcFace", help="Face recognition model (VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace, GhostFaceNet, Buffalo_L)")
    parser.add_argument("--detector", default="retinaface", help="Face detector (retinaface, mtcnn, opencv, dlib)")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold. If not set it is used default for model.")
    args = parser.parse_args()

    extract_and_compare_faces(args.image1, args.image2, model_name=args.model, detector=args.detector, threshold=args.threshold)
