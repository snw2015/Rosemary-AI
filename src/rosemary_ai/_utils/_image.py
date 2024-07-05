from base64 import b64encode

from ..multi_modal.image import Image


def _image_to_base64(img: Image) -> str:
    assert not img.is_url, 'Only support local image'

    with open(img.src, 'rb') as f:
        return b64encode(f.read()).decode()


def _image_to_data_uri(img: Image) -> str:
    if img.is_url:
        return img.src
    else:
        return f'data:{img.mimetype};base64,{_image_to_base64(img)}'
