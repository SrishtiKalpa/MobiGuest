from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

router = APIRouter()

class ScrapeRequest(BaseModel):
    url: str
    max_depth: int = 1

class ScrapeResponse(BaseModel):
    status: str
    message: str
    scraped_content: List[str] = []
    visited_urls: List[str] = []

@router.post("/scrape", response_model=ScrapeResponse)
async def scrape_website(request: ScrapeRequest):
    """
    Scrape a website and return its content.
    """
    try:
        visited_urls = set()
        scraped_content = []
        
        def _scrape(url: str, current_depth: int):
            if current_depth > request.max_depth or url in visited_urls:
                return []
                
            visited_urls.add(url)
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract text content
                body_text = soup.body.get_text(separator=' ', strip=True) if soup.body else ''
                if body_text:
                    scraped_content.append(body_text)
                    
                # If we haven't reached max depth, follow links
                if current_depth < request.max_depth:
                    for link in soup.find_all('a', href=True):
                        next_url = urljoin(url, link['href'])
                        if _is_valid_url(next_url) and next_url not in visited_urls:
                            _scrape(next_url, current_depth + 1)
                            
            except Exception as e:
                print(f"Error scraping {url}: {str(e)}")
                
        # Start scraping
        _scrape(request.url, 0)
        
        return ScrapeResponse(
            status="success",
            message=f"Successfully scraped {len(visited_urls)} pages",
            scraped_content=scraped_content,
            visited_urls=list(visited_urls)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _is_valid_url(url: str) -> bool:
    """Check if a URL is valid for scraping."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
