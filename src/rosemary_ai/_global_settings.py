class GlobalSettings:

    def __init__(self):
        self._settings = {
            'DRY_RUN': False,
        }

    def set(self, key: str, value):
        self._settings[key] = value

    def get(self, key: str):
        return self._settings[key]


SETTINGS = GlobalSettings()
