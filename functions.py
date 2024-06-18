from urllib.parse import urlparse


def extract_path_from_url(full_url):
    parsed_url = urlparse(full_url)
    url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
    return [url, parsed_url.netloc]