import re
from urlparse import urlparse

def parse_s3_url(url):
    """Parses an S3 URL to extract the bucket and key.

    This function handles a few different S3 URL formats:
        http://s3.amazonaws.com/bucket/key
        http://bucket.s3.amazonaws.com/key
        s3://bucket/key

    Args:
        url: The S3 URL to parse.

    Returns a tuple containing (bucket, key). If the provided URL is
    not a valid S3 URL, an empty bucket and key are returned,
    i.e. ("", "").

    """

    # Parse URL
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc
    path = parsed_url.path
    if path.startswith("/"):
        path = path[1:]

    # Initialize empty bucket/key and check the hostname
    bucket = ""
    key = ""
    if netloc == "s3.amazonaws.com":
        # Bucket is the first part of the path
        split_path = path.split("/")
        if len(split_path) >= 2:
            bucket = split_path[0]
            key = "/".join(split_path[1:])
    elif netloc:
        # Hostname is not empty, so bucket is in the netloc as a
        # "virtual host"
        m_obj = re.search(r"([^\.]+)", netloc)
        if m_obj is not None:
            bucket = m_obj.group(1)
            key = path

    return (bucket, key)

def is_s3_url(url):
    """Returns whether a URL is an S3 path or not.

    Args:
        url: The path to check.

    """
    (bucket, key) = parse_s3_url(url)
    return (bucket and key)

