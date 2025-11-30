import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

def scrape_website(url, max_depth, current_depth=0):
    if current_depth > max_depth or url in st.session_state.visited_urls:
        return []

    st.session_state.visited_urls.add(url)
    scraped_content = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
        if body_text:
            scraped_content.append(body_text)

        if current_depth < max_depth:
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(url, href)
                
                if urlparse(absolute_url).netloc == urlparse(url).netloc:
                    scraped_content.extend(scrape_website(absolute_url, max_depth, current_depth + 1))

    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
    except Exception as e:
        st.warning(f"An error occurred while processing {url}: {e}")

    return scraped_content
