import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from scipy.optimize import curve_fit
import matplotlib.cm as cm


def enhance_with_clahe_and_bicubic(image_np):
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    upscaled = cv2.resize(enhanced, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    return upscaled


def apply_false_colormap(image_np, colormap='turbo'):
    if len(image_np.shape) == 3:
        image_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = image_np

    image_norm = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    colormap_dict = {
        "turbo": cm.turbo,
        "viridis": cm.viridis,
        "plasma": cm.plasma,
        "jet": cm.jet,
        "inferno": cm.inferno,
        "magma": cm.magma
    }

    if colormap in colormap_dict:
        cmap = colormap_dict[colormap]
        color_mapped = cmap(image_norm / 255.0)
        color_mapped = (color_mapped[:, :, :3] * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported colormap: {colormap}")

    return color_mapped


def extract_thermal_profile(image_np, y_row=None):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if y_row is None:
        y_row = h // 2
    profile = gray[y_row, :]
    return profile


def apply_contour_overlay(image_np, threshold=128):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image_np.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    return contour_image


def apply_gradient_magnitude(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobelx, sobely)
    normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def apply_laplacian_of_gaussian(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    log = gaussian_laplace(gray, sigma=1.5)
    log = np.abs(log)
    log = (log / log.max()) * 255
    return cv2.cvtColor(log.astype(np.uint8), cv2.COLOR_GRAY2RGB)


def apply_heat_anomaly_overlay(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    z_score = (gray - np.mean(gray)) / (np.std(gray) + 1e-5)
    heat = np.clip((z_score * 32) + 128, 0, 255).astype(np.uint8)
    heat_colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_colored, cv2.COLOR_BGR2RGB)


def extract_melt_pool_properties(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
        stats.append({"area": area, "width": w, "height": h, "centroid": (cx, cy)})
    return stats


def two_d_gaussian(coords, amplitude, xo, yo, sigma_x, sigma_y, offset):
    x, y = coords
    xo = float(xo)
    yo = float(yo)
    g = offset + amplitude * np.exp(
        -(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2))
    )
    return g.ravel()


def fit_2d_gaussian(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    x = np.linspace(0, gray.shape[1] - 1, gray.shape[1])
    y = np.linspace(0, gray.shape[0] - 1, gray.shape[0])
    x, y = np.meshgrid(x, y)
    initial_guess = (gray.max(), gray.shape[1] / 2, gray.shape[0] / 2, 10, 10, gray.min())
    try:
        popt, _ = curve_fit(two_d_gaussian, (x, y), gray.ravel(), p0=initial_guess)
        return {
            "amplitude": popt[0],
            "xo": popt[1],
            "yo": popt[2],
            "sigma_x": popt[3],
            "sigma_y": popt[4],
            "offset": popt[5],
            "ellipticity": round(popt[3] / popt[4], 2) if popt[4] != 0 else float('inf')
        }
    except Exception as e:
        return {"error": str(e)}
