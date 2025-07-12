class FrameDecodeError(Exception):
    """Ошибка декодирования кадра"""
    def __init__(self, message="Ошибка декодирования кадра"):
        super().__init__(message)
        self.message = message
