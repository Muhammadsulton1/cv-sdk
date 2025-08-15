class S3Error(Exception):
    pass


class S3DownloadError(S3Error):
    pass


class S3UploadError(S3Error):
    pass


class S3ConnectionError(S3Error):
    pass


class S3BucketError(S3Error):
    pass


class NatsError(Exception):
    pass