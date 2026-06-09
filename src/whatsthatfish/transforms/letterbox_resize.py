from PIL import Image


class LetterboxResize:
    """Resize a PIL image to a square canvas while preserving aspect ratio.

    Scales the image so the longer side equals `size`, then pads the shorter
    side symmetrically with zeros to produce a (size x size) output.
    Odd padding remainder goes to the right/bottom edge.
    """

    def __init__(self, size: int = 320):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left, top = pad_w // 2, pad_h // 2

        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(img, (left, top))
        return canvas
