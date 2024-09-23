from colorsys import hls_to_rgb, rgb_to_hls

import click
import numpy as np
from PIL import Image, ImageColor

from .kmeans import kmeans


def to_hex(color):
    """Convert color to hex string"""

    r, g, b = color
    return f"#{r:02x}{g:02x}{b:02x}"


def blend_colors(color, bg_color, alpha=255):
    """Blend color with background color"""

    alpha /= 255
    bg_color = np.array(bg_color)

    return alpha * np.array(color) + (1 - alpha) * bg_color


def luminance_value(value):
    """Get luminance value from color value"""

    if value > 1:
        value /= 255

    return value / 12.92 if value < 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def color_luminance(rgb_color):
    """Get luminance from rgb color"""

    R, G, B = [luminance_value(val) for val in rgb_color]
    return (0.2126 * R + 0.7152 * G + 0.0722 * B) + 0.05


def contrast_ratio(color1, color2):
    """Get contrast ratio between color1 and color2"""

    luminance1 = color_luminance(color1)
    luminance2 = color_luminance(color2)

    if luminance1 > luminance2:
        return luminance1 / luminance2
    return luminance2 / luminance1


def adjust_closest_color(rgb_color, target_color, target_contrast=3):
    """Adjust rgb_color to have target_contrast contrast ratio with target_color"""

    r, g, b = np.array(rgb_color) / 255
    h, _, s = rgb_to_hls(r, g, b)

    for lum in np.arange(1, 0, -1 / 255):
        color = hls_to_rgb(h, lum, s)
        ratio = contrast_ratio(color, target_color)
        if np.isclose(ratio, target_contrast, rtol=5e-2):
            break

    # scale rgb values to 0-255 range
    closest_color = np.clip(color, 0, 1) * 255
    return closest_color.astype(np.uint8)


def get_highest_contrast_color(target_color, candidate_colors, target_contrast=3):
    """Get the first color from candidate_colors with contrast ratio >= target_contrast
    if no color is found, adjust the color with highest contrast_ratio"""

    contrast_ratios = []
    for color in candidate_colors:
        ratio = contrast_ratio(target_color, color)
        if ratio >= target_contrast:
            return color
        contrast_ratios.append(ratio)

    # adjust color with highest contrast_ratio
    index = np.argmax(contrast_ratios)
    return adjust_closest_color(candidate_colors[index], target_color, target_contrast)


def preprocess_image(image):
    """Preprocess image get unique hsv colors and their frequencies,
    desaturating colors and ignoring the darkest and less frequent colors"""

    width, height = image.size
    image_hsv = np.array(image.resize((width // 2, height // 2)).convert("HSV")).reshape(-1, 3)

    # ignore darkest colors
    image_hsv = image_hsv[image_hsv[:, 2] > 25]

    # desaturate colors
    image_hsv[:, 1] = np.minimum(image_hsv[:, 1], 200)

    # ignore less frequent colors
    unique_colors, freqs = np.unique(image_hsv, axis=0, return_counts=True)

    unique_colors = unique_colors[freqs > 4]
    freqs = freqs[freqs > 4]

    return unique_colors, freqs / freqs.sum()


def color_extractor(image_path, target_contrast, bg_popup, alpha, bg_topbar):
    # convert color strings to rgb tuples
    bg_popup = ImageColor.getrgb(bg_popup)
    bg_topbar = ImageColor.getrgb(bg_topbar)

    image = Image.open(image_path)
    unique_colors, freqs = preprocess_image(image)

    # get dominant colors from image
    dominant_colors_hsv = kmeans(data=unique_colors, sample_weights=freqs, n_clusters=5)
    dominant_colors = Image.fromarray(dominant_colors_hsv.reshape(1, -1, 3), "HSV").convert("RGB")
    dominant_colors = np.array(dominant_colors, dtype=np.uint8).reshape(-1, 3)

    most_dominant_color, *candidate_colors = dominant_colors

    click.echo(to_hex(most_dominant_color))

    # blend most dominant color with background
    bg_blended = blend_colors(most_dominant_color, bg_popup, alpha)
    fg_color = get_highest_contrast_color(
        bg_blended, candidate_colors, target_contrast=target_contrast
    )

    click.echo(to_hex(fg_color))

    # get mediabar progress border color
    mediabar_progress_color = get_highest_contrast_color(
        bg_topbar, [most_dominant_color], target_contrast
    )
    click.echo(to_hex(mediabar_progress_color))
