import mimetypes

mimetype_map = mimetypes.types_map


class Image:
    def __init__(self, src: str, metadata=None):
        if metadata is None:
            metadata = {}
        self.src = src.strip()
        self.is_url = self.src.startswith('http')
        self.mimetype = mimetype_map.get('.' + src.split('.')[-1], 'image/jpeg')
        self.metadata = metadata

    def __str__(self):
        return f'Image(src={self.src}, metadata={self.metadata})'

    def __repr__(self):
        return self.__str__()
