import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

visited_urls = set()
file_name = 'visited_urls.txt'

def is_valid_url(url, domain):
    parsed = urlparse(url)
    return parsed.scheme in ('http', 'https') and domain in parsed.netloc

def get_all_links(url, domain):
    links = set()
    try:
        response = requests.get(url)
        # print(response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        # print(soup.find_all('a'))
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(url, href)
            if is_valid_url(full_url, domain):
                links.add(full_url)
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
    return links


def create_file(links):

    # Write the links to the file
    print("Total links from docs.kommunitas.net: "+str(len(links)))
    with open(file_name, 'w') as f:
        for link in links:
            f.write(link + '\n')

    print(f"Links written to '{file_name}' successfully.")

from collections import deque

def crawl(url, domain, limit=5000):
    q = deque([])
    q.append(url)
    visited = set()
    seen = set()
    count = 1
    skips = 0
    while len(visited) <= limit:
        try:
            current = q.popleft().strip().strip('/').lower().split('#')[0]
        except IndexError:
            break
        if current in visited:
            skips += 1
            continue
        visited.add(current)
        print(f"len: {len(q)} skips: {skips}, {count}) {current}")
        count += 1
        links = get_all_links(current, domain)
        for link in links:
            refined_link = link.strip().strip('/').lower().split('#')[0]
            if refined_link not in visited and refined_link not in seen and domain in refined_link:
                q.append(refined_link)
                seen.add(refined_link)
            else:
              skips += 1
    return list(visited)