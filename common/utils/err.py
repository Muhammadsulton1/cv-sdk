class S3Error(Exception):
    pass


class S3DownloadError(S3Error):
    pass


class S3FrameDecodeError(S3Error):
    pass


class S3UploadError(S3Error):
    pass


class S3ConnectionError(S3Error):
    pass


class S3GetError(S3Error):
    pass


class S3CreateBucketError(S3Error):
    pass


class NatsError(Exception):
    pass


class NatsConnectionError(NatsError):
    pass


class StreamError(Exception):
    pass


class StreamDecodeError(StreamError):
    pass


class StreamFinishedError(StreamError):
    pass
