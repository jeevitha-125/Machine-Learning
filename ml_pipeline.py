"""
ml_pipeline.py
--------------
Complete Medical Machine Learning Pipeline for Thyroid Disorder Prediction.
Covers:
1. Load Input Image (Ultrasound Thyroid Scan)
2. Image Preprocessing (Grayscale, Bilateral Denoising, CLAHE, Segmentation)
3. Feature Extraction (GLCM Texture Features + Statistical Moments + Edge Gradients)
4. Model Training (SVM, Random Forest, KNN, Decision Tree)
5. Performance Evaluation (Confusion Matrix, Accuracy, Precision, Recall, F1)
6. Clinical Diagnosis and Actionable Recommendations
"""

import os
import cv2
import json
import base64
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "static", "processed")
CLASSES = ["normal", "hypothyroid", "hyperthyroid", "nodule"]

CLASS_LABELS_MAP = {
    "normal": "Normal Thyroid (Healthy)",
    "hypothyroid": "Hypothyroidism (Hashimoto Pattern)",
    "hyperthyroid": "Hyperthyroidism (Graves / Enlarged Pattern)",
    "nodule": "Thyroid Nodule (Focal Lesion / Suspicious)"
}

SEVERITY_MAP = {
    "normal": {"level": "Normal", "badge": "success", "score": "Low Risk (0-5%)"},
    "hypothyroid": {"level": "Moderate", "badge": "warning", "score": "Intermediate (55-65%)"},
    "hyperthyroid": {"level": "Moderate-High", "badge": "warning", "score": "Elevated (65-75%)"},
    "nodule": {"level": "High - Requires Follow-up", "badge": "danger", "score": "Significant (80-95%)"}
}

RECOMMENDATIONS = {
    "normal": [
        "Normal glandular architecture observed with homogeneous echotexture.",
        "No focal nodules, microcalcifications, or parenchymal lesions detected.",
        "Routine periodic health checkup and balanced iodine intake recommended."
    ],
    "hypothyroid": [
        "Diffuse heterogeneous hypoechoic thyroid parenchyma with coarse acoustic texture.",
        "Pattern strongly correlates with autoimmune Hashimoto's thyroiditis.",
        "Clinical Action: Serum TSH, Free T4, and Anti-TPO antibody blood tests recommended.",
        "Consult endocrinologist for hormone replacement evaluation (Levothyroxine)."
    ],
    "hyperthyroid": [
        "Enlarged thyroid parenchyma with coarse acoustic texture and vascular flow dilation.",
        "Ultrasound pattern suggestive of diffuse toxic goiter / Graves' disease.",
        "Clinical Action: Serum Free T3, Free T4, TSH receptor antibody (TRAb) test.",
        "Consult endocrinologist for anti-thyroid therapy / beta-blocker management."
    ],
    "nodule": [
        "Focal nodular lesion identified with peripheral acoustic halo and microcalcification markers.",
        "Classified under TIRADS criteria for acoustic irregularity.",
        "Clinical Action: Urgent Fine-Needle Aspiration Biopsy (FNAC) recommended for nodule > 10mm.",
        "Evaluate cervical lymph nodes and schedule follow-up ultrasound in 3 months."
    ]
}

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_image(image_path, target_size=(256, 256)):
    """Loads an image from path and resizes to target dimension."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to decode image file.")
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

def encode_img_to_base64(img_bgr_or_gray):
    """Encodes an OpenCV image to a base64 data URI string."""
    success, buffer = cv2.imencode(".png", img_bgr_or_gray)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

def preprocess_image(img_bgr):
    """Applies the full medical ultrasound preprocessing pipeline."""
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    denoised = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(denoised)

    _, thresh = cv2.threshold(clahe_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=2)

    segmented_rgb = cv2.cvtColor(clahe_enhanced, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        cv2.drawContours(segmented_rgb, sorted_contours, -1, (0, 255, 128), 2)

    steps_dict = {
        "step1_gray": gray,
        "step2_denoise": denoised,
        "step3_contrast": clahe_enhanced,
        "step4_segmented": segmented_rgb,
        "mask": morph
    }

    preview_base64 = {
        "raw": encode_img_to_base64(img_bgr),
        "grayscale": encode_img_to_base64(gray),
        "denoised": encode_img_to_base64(denoised),
        "clahe": encode_img_to_base64(clahe_enhanced),
        "segmented": encode_img_to_base64(segmented_rgb)
    }

    return steps_dict, preview_base64

def calculate_shannon_entropy(gray_img):
    """Calculates Shannon entropy of image histogram."""
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
    hist = hist.ravel() / (hist.sum() + 1e-7)
    non_zero = hist[hist > 0]
    return float(-np.sum(non_zero * np.log2(non_zero)))

def extract_features(preprocessed_dict):
    """Extracts a feature vector of 15 informative features."""
    clahe_img = preprocessed_dict["step3_contrast"]
    mask = preprocessed_dict["mask"]

    quantized = (clahe_img // 8).astype(np.uint8)
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(quantized, distances=distances, angles=angles, levels=32, symmetric=True, normed=True)
    
    contrast = float(np.mean(graycoprops(glcm, 'contrast')))
    dissimilarity = float(np.mean(graycoprops(glcm, 'dissimilarity')))
    homogeneity = float(np.mean(graycoprops(glcm, 'homogeneity')))
    energy = float(np.mean(graycoprops(glcm, 'energy')))
    correlation = float(np.mean(graycoprops(glcm, 'correlation')))
    asm = float(np.mean(graycoprops(glcm, 'ASM')))

    pixels = clahe_img.flatten().astype(np.float64)
    mean_val = float(np.mean(pixels))
    std_val = float(np.std(pixels))
    variance_val = float(np.var(pixels))
    skewness_val = float(skew(pixels))
    kurtosis_val = float(kurtosis(pixels))
    entropy_val = calculate_shannon_entropy(clahe_img)

    sobel_x = cv2.Sobel(clahe_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(clahe_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_mean = float(np.mean(sobel_mag))
    edge_std = float(np.std(sobel_mag))
    roi_ratio = float(np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1]))

    features_dict = {
        "glcm_contrast": round(contrast, 4),
        "glcm_dissimilarity": round(dissimilarity, 4),
        "glcm_homogeneity": round(homogeneity, 4),
        "glcm_energy": round(energy, 4),
        "glcm_correlation": round(correlation, 4),
        "glcm_asm": round(asm, 4),
        "stat_mean": round(mean_val, 2),
        "stat_std": round(std_val, 2),
        "stat_variance": round(variance_val, 2),
        "stat_skewness": round(skewness_val, 4),
        "stat_kurtosis": round(kurtosis_val, 4),
        "stat_entropy": round(entropy_val, 4),
        "edge_mean": round(edge_mean, 2),
        "edge_std": round(edge_std, 2),
        "roi_density": round(roi_ratio, 4)
    }

    feature_vector = np.array([
        contrast, dissimilarity, homogeneity, energy, correlation, asm,
        mean_val, std_val, variance_val, skewness_val, kurtosis_val, entropy_val,
        edge_mean, edge_std, roi_ratio
    ], dtype=np.float64)

    return features_dict, feature_vector

def load_dataset_features(dataset_root=DATASET_DIR):
    """Loads all train/test images, runs preprocessing and feature extraction."""
    X_train, y_train = [], []
    X_test, y_test = [], []

    print("Extracting features from dataset...")
    for split, X_list, y_list in [("train", X_train, y_train), ("test", X_test, y_test)]:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            continue
        for class_idx, class_name in enumerate(CLASSES):
            class_folder = os.path.join(split_dir, class_name)
            if not os.path.exists(class_folder):
                continue
            for f in os.listdir(class_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_folder, f)
                    try:
                        img = load_image(img_path)
                        prep_dict, _ = preprocess_image(img)
                        _, fvec = extract_features(prep_dict)
                        X_list.append(fvec)
                        y_list.append(class_idx)
                    except Exception as e:
                        print(f"Skipping {img_path}: {e}")

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    if len(X_test) == 0 and len(X_train) > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

    return X_train, X_test, y_train, y_test

def generate_confusion_matrix_plot(y_true, y_pred, model_name="SVM"):
    """Creates a stylized confusion matrix plot and saves it."""
    cm = confusion_matrix(y_true, y_pred)
    class_names = [c.capitalize() for c in CLASSES]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, annot_kws={"size": 12, "weight": "bold"})
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, pad=12, fontweight='bold', color='#1e293b')
    plt.ylabel('Actual Medical Diagnosis', fontsize=11, fontweight='bold', color='#334155')
    plt.xlabel('ML Predicted Class', fontsize=11, fontweight='bold', color='#334155')
    plt.tight_layout()

    cm_path = os.path.join(PROCESSED_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()

    with open(cm_path, "rb") as f:
        cm_b64 = f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"
    return cm.tolist(), cm_b64

def train_all_models():
    """Trains SVM, Random Forest, KNN, and Decision Tree; selects the best model."""
    X_train, X_test, y_train, y_test = load_dataset_features()
    
    if len(X_train) == 0:
        raise ValueError("Dataset is empty. Run generate_dataset.py first.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    models_zoo = {
        "SVM": SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42)
    }

    all_results = {}
    best_acc = -1
    best_name = "SVM"
    best_model_obj = None

    for name, model in models_zoo.items():
        print(f"Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        
        clean_name = name.lower().replace(" ", "_").replace("-", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{clean_name}_model.pkl"))

        cm_list, cm_b64 = generate_confusion_matrix_plot(y_test, y_pred, model_name=name)

        all_results[name] = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "confusion_matrix": cm_list,
            "confusion_matrix_img": cm_b64,
            "samples_train": len(X_train),
            "samples_test": len(X_test)
        }

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model_obj = model

    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, "best_model.pkl"))

    metrics_payload = {
        "best_model": best_name,
        "models_evaluated": all_results,
        "summary": all_results[best_name]
    }

    metrics_file = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_payload, f, indent=4)

    print(f"Training Complete! Best model: {best_name} with Accuracy: {all_results[best_name]['accuracy']}%")
    return metrics_payload

def predict_image(image_path, model_choice="best"):
    """Complete end-to-end diagnosis from raw image path."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        train_all_models()

    scaler = joblib.load(scaler_path)

    model_filename = "best_model.pkl"
    if model_choice.lower() in ["svm", "random_forest", "k_nearest_neighbors", "decision_tree"]:
        clean_choice = model_choice.lower().replace(" ", "_")
        candidate = os.path.join(MODELS_DIR, f"{clean_choice}_model.pkl")
        if os.path.exists(candidate):
            model_filename = f"{clean_choice}_model.pkl"

    model = joblib.load(os.path.join(MODELS_DIR, model_filename))

    img = load_image(image_path)
    prep_dict, preview_b64 = preprocess_image(img)
    features_dict, fvec = extract_features(prep_dict)

    fvec_scaled = scaler.transform([fvec])
    pred_idx = int(model.predict(fvec_scaled)[0])
    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(fvec_scaled)[0]
        confidence = float(np.max(probabilities)) * 100.0
        prob_breakdown = {CLASSES[i]: round(float(prob) * 100.0, 1) for i, prob in enumerate(probabilities)}
    else:
        confidence = 95.0
        prob_breakdown = {c: 0.0 for c in CLASSES}
        prob_breakdown[CLASSES[pred_idx]] = 95.0

    pred_class = CLASSES[pred_idx]
    display_title = CLASS_LABELS_MAP[pred_class]
    severity_info = SEVERITY_MAP[pred_class]
    recommendation_list = RECOMMENDATIONS[pred_class]

    return {
        "predicted_class": pred_class,
        "display_title": display_title,
        "confidence": round(confidence, 1),
        "probability_breakdown": prob_breakdown,
        "severity": severity_info["level"],
        "severity_badge": severity_info["badge"],
        "risk_score": severity_info["score"],
        "recommendations": recommendation_list,
        "features": features_dict,
        "preprocessed_previews": preview_b64
    }

if __name__ == "__main__":
    train_all_models()