"""
Web scraper for extracting content from documentation sites.

This module handles:
- Static HTML scraping
- JavaScript-rendered content (using Selenium)
- Content extraction and cleaning
- Metadata collection (title, URL, date)
"""

import re
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException

from utils.logger import logger


class WebScraper:
    """Scraper for documentation websites."""
    
    def __init__(self, use_selenium: bool = False, headless: bool = True):
        """
        Initialize web scraper.
        
        Args:
            use_selenium: Whether to use Selenium for JS-rendered content
            headless: Run browser in headless mode
        """
        self.use_selenium = use_selenium
        self.driver = None
        
        if use_selenium:
            try:
                chrome_options = Options()
                if headless:
                    chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                
                self.driver = webdriver.Chrome(options=chrome_options)
                logger.info("Selenium WebDriver initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Selenium: {e}. Falling back to requests only.")
                self.use_selenium = False
    
    def scrape_url(self, url: str) -> Dict[str, any]:
        """
        Scrape a single URL and extract content.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with content, title, url, and metadata
        """
        logger.info(f"Scraping URL: {url}")
        
        try:
            if self.use_selenium and self.driver:
                content = self._scrape_with_selenium(url)
            else:
                content = self._scrape_with_requests(url)
            
            if not content:
                logger.warning(f"No content extracted from {url}")
                return None
            
            # Extract metadata
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract title
            title = self._extract_title(soup)
            
            # Extract main content
            text_content = self._extract_text(soup)
            
            # Extract date if available
            date = self._extract_date(soup, url)
            
            return {
                "url": url,
                "title": title,
                "content": text_content,
                "date": date,
                "scraped_at": datetime.now().isoformat(),
                "metadata": {
                    "url": url,
                    "domain": urlparse(url).netloc
                }
            }
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return None
    
    def scrape_multiple_urls(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped content dictionaries
        """
        results = []
        for url in urls:
            result = self.scrape_url(url)
            if result:
                results.append(result)
        return results
    
    def _scrape_with_requests(self, url: str) -> Optional[str]:
        """Scrape using requests library (faster, but no JS support)."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error(f"Requests scraping failed for {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape using Selenium (slower, but supports JavaScript)."""
        try:
            self.driver.get(url)
            self.driver.implicitly_wait(3)
            return self.driver.page_source
        except TimeoutException:
            logger.error(f"Selenium timeout for {url}")
            return None
        except Exception as e:
            logger.error(f"Selenium scraping failed for {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try multiple selectors
        title_selectors = [
            ('h1', {}),
            ('title', {}),
            ('meta[property="og:title"]', {'content': True}),
        ]
        
        for selector, kwargs in title_selectors:
            element = soup.select_one(selector)
            if element:
                if kwargs.get('content'):
                    return element.get('content', '').strip()
                return element.get_text().strip()
        
        return "Untitled"
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract main text content from page."""
        # Try to find main content area
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '#main'
        ]
        
        main_content = None
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if not main_content:
            main_content = soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            # Clean up excessive whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text.strip()
        
        return ""
    
    def _extract_date(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Extract publication/update date if available."""
        date_selectors = [
            'time[datetime]',
            'meta[property="article:published_time"]',
            'meta[name="date"]',
            'meta[name="publishdate"]',
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                date_str = element.get('datetime') or element.get('content')
                if date_str:
                    return date_str
        
        # Try to extract from URL if it contains date
        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', url)
        if date_match:
            return date_match.group(0)
        
        return None
    
    def save_scraped_content(self, content: Dict[str, any], output_dir: Path):
        """Save scraped content to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from URL
        url_path = urlparse(content['url']).path
        filename = url_path.replace('/', '_').strip('_') or 'index'
        filename = re.sub(r'[^\w\-_]', '_', filename)
        
        filepath = output_dir / f"{filename}.json"
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved scraped content to {filepath}")
    
    def __del__(self):
        """Cleanup: close Selenium driver if open."""
        if self.driver:
            self.driver.quit()


class ContentExtractor:
    """Extract and clean content from various sources."""
    
    @staticmethod
    def extract_from_file(filepath: Path) -> Optional[Dict[str, any]]:
        """
        Extract content from a local file.
        
        Supports: .txt, .md, .html
        """
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return None
        
        suffix = filepath.suffix.lower()
        
        if suffix == '.txt' or suffix == '.md':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return {
                "url": f"file://{filepath.absolute()}",
                "title": filepath.stem,
                "content": content,
                "date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                "scraped_at": datetime.now().isoformat(),
                "metadata": {
                    "filepath": str(filepath),
                    "extension": suffix
                }
            }
        
        elif suffix == '.html':
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style
            for script in soup(["script", "style"]):
                script.decompose()
            
            title = soup.find('title')
            title = title.get_text().strip() if title else filepath.stem
            
            text_content = soup.get_text(separator='\n', strip=True)
            text_content = re.sub(r'\n{3,}', '\n\n', text_content)
            
            return {
                "url": f"file://{filepath.absolute()}",
                "title": title,
                "content": text_content,
                "date": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
                "scraped_at": datetime.now().isoformat(),
                "metadata": {
                    "filepath": str(filepath),
                    "extension": suffix
                }
            }
        
        else:
            logger.warning(f"Unsupported file type: {suffix}")
            return None
