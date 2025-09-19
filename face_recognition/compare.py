import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import numpy as np
import cv2
from deepface import DeepFace
from deepface.modules import verification
from deepface.models.FacialRecognition import FacialRecognition
from scipy.spatial.distance import cosine, cdist
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


def resize_with_aspect_ratio(image, max_size=1024):
    scale = max_size / max(image.height, image.width)
    return image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS
    ), scale

def detect_faces(image_array, detector_backend="retinaface"):
    return DeepFace.extract_faces(
        image_array,
        detector_backend=detector_backend,
        enforce_detection=False
    )

def get_embeddings(model, faces, target_size):
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


def one_to_one_matching(faces1, faces2, model, target_size, metric, threshold):
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


def draw_id_with_background(draw, x, y, w, h, text, font, text_color="black",
                            bg_color="white", border=FACE_BORDER, padding=2, vmargin=2,
                            image_height=None):
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


def extract_and_compare_faces(image1_path, image2_path, model_name="ArcFace", detector="retinaface", use_default_threshold=False, custom_threshold=0.5):
    if not os.path.isfile(image1_path) or not os.path.isfile(image2_path):
        print("Error: both image files are required.")
        return

    print("Loading images...")
    img1 = Image.open(image1_path).convert("RGB")
    img2 = Image.open(image2_path).convert("RGB")

    print(f"Detecting faces using '{detector}' detector...")
    faces1 = detect_faces(np.array(img1), detector)
    faces2 = detect_faces(np.array(img2), detector)

    print(f"Recognizing faces using '{model_name}'model...")
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
    threshold = default_threshold if use_default_threshold else custom_threshold

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
        color = "green" if distance < min(custom_threshold, default_threshold) else "yellow"
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
            color = "\033[92m" if distance < min(custom_threshold, default_threshold) else "\033[93m"
            print(f"{color}Face ID {i} of {image1_path} matches face ID {j} of {image2_path} (distance: {distance:.3f}){reset}")
    else:
        print("No matching faces found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFace comparison of faces found on input images.")
    parser.add_argument("image1", help="Path to the 1st image")
    parser.add_argument("image2", help="Path to the 2nd image")
    parser.add_argument("--model", default="ArcFace", help="Face recognition model (VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace, GhostFaceNet, Buffalo_L)")
    parser.add_argument("--detector", default="retinaface", help="Face detector (retinaface, mtcnn, opencv, dlib)")
    parser.add_argument("--use-default-threshold", action="store_true", help="Use DeepFace default threshold instead of custom one")
    parser.add_argument("--threshold", type=float, default=0.5, help="Custom threshold (ignored if --use-default-threshold is set)")
    args = parser.parse_args()

    extract_and_compare_faces(args.image1, args.image2, model_name=args.model, detector=args.detector,
        use_default_threshold=args.use_default_threshold, custom_threshold=args.threshold)
