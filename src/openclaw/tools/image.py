"""Image processing and analysis tool — resize, convert, crop, and inspect images."""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from langchain_core.tools import tool

IMAGES_DIR = Path.home() / ".langclaw" / "images"
DOWNLOADS_DIR = Path.home() / ".langclaw" / "browser" / "downloads"


def _ensure_dirs():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _download_image(url: str) -> Path:
    """Download an image from a URL and return the local path."""
    import httpx

    _ensure_dirs()
    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    # Derive filename from URL or content-type
    content_type = response.headers.get("content-type", "")
    ext = ".png"
    if "jpeg" in content_type or "jpg" in content_type:
        ext = ".jpg"
    elif "webp" in content_type:
        ext = ".webp"
    elif "gif" in content_type:
        ext = ".gif"
    elif "bmp" in content_type:
        ext = ".bmp"
    elif "png" in content_type:
        ext = ".png"
    else:
        # Try to get extension from URL
        url_path = url.split("?")[0].split("#")[0]
        if "." in url_path.split("/")[-1]:
            url_ext = "." + url_path.split("/")[-1].rsplit(".", 1)[-1].lower()
            if url_ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
                ext = url_ext

    filename = f"download_{uuid.uuid4().hex[:8]}{ext}"
    dest = DOWNLOADS_DIR / filename
    dest.write_bytes(response.content)
    return dest


def _auto_output_path(original_path: Path, suffix: str, output_format: str = "") -> Path:
    """Generate an automatic output path in the images directory."""
    _ensure_dirs()
    stem = original_path.stem
    ext = f".{output_format.lower()}" if output_format else original_path.suffix
    filename = f"{stem}_{suffix}_{uuid.uuid4().hex[:6]}{ext}"
    return IMAGES_DIR / filename


def _format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _describe_image(img, img_path: Path) -> str:
    """Return a structured description of the image with metadata and EXIF data."""
    from PIL.ExifTags import TAGS

    file_size = img_path.stat().st_size
    lines = [
        f"Image: {img_path.name}",
        f"  Path: {img_path}",
        f"  Format: {img.format or 'unknown'}",
        f"  Mode: {img.mode}",
        f"  Size: {img.width} x {img.height} pixels",
        f"  File size: {_format_file_size(file_size)}",
    ]

    # Extract EXIF data if available
    exif_data = {}
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                # Skip binary or overly long values
                str_val = str(value)
                if len(str_val) > 200:
                    str_val = str_val[:200] + "..."
                exif_data[tag_name] = str_val
    except Exception:
        pass

    if exif_data:
        lines.append("  EXIF data:")
        for key, val in sorted(exif_data.items()):
            lines.append(f"    {key}: {val}")
    else:
        lines.append("  EXIF data: none")

    return "\n".join(lines)


def _info_image(img, img_path: Path) -> str:
    """Return structured metadata about the image."""
    file_size = img_path.stat().st_size
    has_alpha = img.mode in ("RGBA", "LA", "PA")

    # Determine color depth
    mode_to_depth = {
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 24,
        "RGBA": 32,
        "CMYK": 32,
        "YCbCr": 24,
        "LAB": 24,
        "HSV": 24,
        "I": 32,
        "F": 32,
        "LA": 16,
        "PA": 16,
    }
    color_depth = mode_to_depth.get(img.mode, 0)

    lines = [
        f"Image info: {img_path.name}",
        f"  dimensions: {img.width} x {img.height}",
        f"  format: {img.format or 'unknown'}",
        f"  mode: {img.mode}",
        f"  color_depth: {color_depth} bits",
        f"  file_size: {_format_file_size(file_size)}",
        f"  has_alpha: {has_alpha}",
    ]
    return "\n".join(lines)


def _resize_image(img, img_path: Path, width: int, height: int, output_path: str) -> str:
    """Resize the image, maintaining aspect ratio if only one dimension given."""
    from PIL import Image

    orig_w, orig_h = img.size

    if width > 0 and height > 0:
        new_size = (width, height)
    elif width > 0:
        ratio = width / orig_w
        new_size = (width, int(orig_h * ratio))
    elif height > 0:
        ratio = height / orig_h
        new_size = (int(orig_w * ratio), height)
    else:
        return "[error] Provide at least width or height for resize."

    resized = img.resize(new_size, Image.LANCZOS)

    dest = Path(output_path) if output_path else _auto_output_path(img_path, "resized")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resized.save(dest)

    return (
        f"Resized {img_path.name}: {orig_w}x{orig_h} -> {new_size[0]}x{new_size[1]}\n"
        f"Saved to: {dest}"
    )


def _convert_image(img, img_path: Path, output_format: str, output_path: str) -> str:
    """Convert image to a different format."""
    fmt = output_format.upper()
    format_map = {
        "JPG": "JPEG",
        "JPEG": "JPEG",
        "PNG": "PNG",
        "WEBP": "WEBP",
        "GIF": "GIF",
        "BMP": "BMP",
    }

    if fmt not in format_map:
        return f"[error] Unsupported format: {output_format}. Supported: png, jpg, webp, gif, bmp"

    save_format = format_map[fmt]

    # Handle RGB/RGBA conversion
    converted = img
    if save_format == "JPEG" and img.mode in ("RGBA", "LA", "PA", "P"):
        # JPEG doesn't support transparency, convert to RGB
        if img.mode == "P":
            converted = img.convert("RGBA")
        if converted.mode in ("RGBA", "LA", "PA"):
            # Composite onto white background
            from PIL import Image

            background = Image.new("RGB", converted.size, (255, 255, 255))
            if converted.mode == "RGBA":
                background.paste(converted, mask=converted.split()[3])
            else:
                background.paste(converted)
            converted = background
        else:
            converted = converted.convert("RGB")
    elif save_format in ("PNG", "WEBP") and img.mode == "CMYK":
        converted = img.convert("RGB")

    ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif", "BMP": ".bmp"}
    ext = ext_map[save_format]

    dest = Path(output_path) if output_path else _auto_output_path(img_path, "converted", ext.lstrip("."))
    dest.parent.mkdir(parents=True, exist_ok=True)
    converted.save(dest, format=save_format)

    return (
        f"Converted {img_path.name} ({img.format or 'unknown'}) -> {save_format}\n"
        f"Saved to: {dest}"
    )


def _thumbnail_image(img, img_path: Path, width: int, height: int, output_path: str) -> str:
    """Create a thumbnail with maximum dimensions."""
    max_w = width if width > 0 else 256
    max_h = height if height > 0 else 256

    from PIL import Image

    thumb = img.copy()
    thumb.thumbnail((max_w, max_h), Image.LANCZOS)

    dest = Path(output_path) if output_path else _auto_output_path(img_path, "thumb")
    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(dest)

    return (
        f"Thumbnail created: {img.width}x{img.height} -> {thumb.width}x{thumb.height}\n"
        f"Max dimensions: {max_w}x{max_h}\n"
        f"Saved to: {dest}"
    )


def _crop_image(img, img_path: Path, crop_box: str, output_path: str) -> str:
    """Crop the image using left,top,right,bottom coordinates."""
    try:
        parts = [int(x.strip()) for x in crop_box.split(",")]
        if len(parts) != 4:
            raise ValueError("Expected 4 values")
        left, top, right, bottom = parts
    except (ValueError, TypeError) as e:
        return f'[error] Invalid crop_box format. Use "left,top,right,bottom" (e.g. "10,20,200,300"). {e}'

    if left >= right or top >= bottom:
        return "[error] Invalid crop coordinates: left must be < right and top must be < bottom."
    if left < 0 or top < 0 or right > img.width or bottom > img.height:
        return (
            f"[error] Crop coordinates out of bounds. "
            f"Image size is {img.width}x{img.height}, "
            f"but got crop box ({left},{top},{right},{bottom})."
        )

    cropped = img.crop((left, top, right, bottom))

    dest = Path(output_path) if output_path else _auto_output_path(img_path, "cropped")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dest)

    return (
        f"Cropped {img_path.name}: ({left},{top}) to ({right},{bottom})\n"
        f"New size: {cropped.width}x{cropped.height}\n"
        f"Saved to: {dest}"
    )


@tool
def image_tool(
    action: str,
    path: str = "",
    url: str = "",
    width: int = 0,
    height: int = 0,
    output_format: str = "",
    output_path: str = "",
    crop_box: str = "",
) -> str:
    """Process or analyze an image file.

    Supports image inspection, resizing, format conversion, thumbnail
    creation, and cropping via Pillow.

    Args:
        action: Operation to perform. One of: "describe", "resize",
                "convert", "info", "thumbnail", "crop".
        path: Path to the image file on disk.
        url: URL to download an image from (alternative to path).
             Downloaded images are saved to ~/.langclaw/browser/downloads/.
        width: Target width in pixels (for resize/thumbnail).
        height: Target height in pixels (for resize/thumbnail).
        output_format: Target format for convert (png, jpg, webp, gif, bmp).
        output_path: Where to save the result. Auto-generated under
                     ~/.langclaw/images/ if empty.
        crop_box: Crop coordinates as "left,top,right,bottom".

    Returns:
        Result description or error message.
    """
    try:
        from PIL import Image
    except ImportError:
        return (
            "[error] Pillow is required for image operations. "
            "Install it with: pip install Pillow"
        )

    valid_actions = ("describe", "resize", "convert", "info", "thumbnail", "crop")
    if action not in valid_actions:
        return f"[error] Unknown action: {action}. Must be one of: {', '.join(valid_actions)}"

    # Resolve the image source
    try:
        if url and not path:
            try:
                img_path = _download_image(url)
            except ImportError:
                return "[error] httpx is required to download images. Install it with: pip install httpx"
            except Exception as e:
                return f"[error] Failed to download image from {url}: {e}"
        elif path:
            img_path = Path(path).expanduser().resolve()
        else:
            return "[error] Provide either 'path' or 'url' for the image."

        if not img_path.exists():
            return f"[error] Image file not found: {img_path}"

        img = Image.open(img_path)

        if action == "describe":
            return _describe_image(img, img_path)
        elif action == "info":
            return _info_image(img, img_path)
        elif action == "resize":
            return _resize_image(img, img_path, width, height, output_path)
        elif action == "convert":
            if not output_format:
                return "[error] 'output_format' is required for convert (e.g. png, jpg, webp)."
            return _convert_image(img, img_path, output_format, output_path)
        elif action == "thumbnail":
            return _thumbnail_image(img, img_path, width, height, output_path)
        elif action == "crop":
            if not crop_box:
                return "[error] 'crop_box' is required for crop (e.g. \"10,20,200,300\")."
            return _crop_image(img, img_path, crop_box, output_path)
        else:
            return f"[error] Unhandled action: {action}"

    except Exception as e:
        return f"[error] Image operation failed: {e}"
