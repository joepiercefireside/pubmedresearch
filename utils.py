import requests
import xml.etree.ElementTree as ET
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import json
from openai import OpenAI
import os
import re
from datetime import datetime, timedelta
import tenacity
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BIOMEDICAL_VOCAB = {
    "diabetes": ["Diabetes Mellitus", "insulin resistance", "type 2 diabetes"],
    "weight loss": ["obesity", "body weight reduction", "fat loss"],
    "treatment": ["therapy", "intervention", "management"],
    "disease": ["disorder", "condition", "pathology"],
    "statins": ["HMG-CoA reductase inhibitors", "atorvastatin", "simvastatin"],
    "heart disease": ["cardiovascular disease", "coronary artery disease", "myocardial infarction"],
    "cardiovascular": ["heart-related", "circulatory"],
    "blood pressure": ["hypertension", "BP"],
    "hypertension": ["high blood pressure", "elevated BP"]
}

@sleep_and_retry
@limits(calls=10, period=1)
@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.info(f"Retrying ESearch, attempt {retry_state.attempt_number}")
)
def esearch(term, db='pubmed', retmax=100, date_range=None, start_year=None, api_key=None):
    base_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={db}&term={term}&retmax={retmax}&retmode=json&sort=relevance"
    if date_range:
        start_date, end_date = date_range.split(':')
        base_url += f"&datetype=pdat&mindate={start_date}&maxdate={end_date}"
    elif start_year:
        base_url += f"&datetype=pdat&mindate={start_year}/01/01"
    if api_key:
        base_url += f"&api_key={api_key}"
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('esearchresult', {}).get('idlist', [])
    except Exception as e:
        logger.error(f"Error in ESearch: {str(e)}")
        raise

@sleep_and_retry
@limits(calls=10, period=1)
def efetch(pmids, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(map(str, pmids)),
        "retmode": "xml",
        "api_key": api_key
    }
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Error in EFetch: {str(e)}")
        raise

def parse_efetch_xml(xml_content):
    try:
        root = ET.fromstring(xml_content)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else "N/A"
            title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No title"
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text or "" if abstract_elem is not None else ""
            authors = [author.find("LastName").text for author in article.findall(".//Author") 
                       if author.find("LastName") is not None]
            journal = article.find(".//Journal/Title")
            journal = journal.text if journal is not None else "N/A"
            pub_date = article.find(".//PubDate/Year")
            pub_date = pub_date.text if pub_date is not None else "N/A"
            logger.info(f"Parsed article: PMID={pmid}, Date={pub_date}, Abstract={'Present' if abstract else 'Missing'}")
            articles.append({
                "title": title,
                "abstract": abstract,
                "authors": ", ".join(authors) if authors else "N/A",
                "journal": journal,
                "publication_date": pub_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        return articles
    except Exception as e:
        logger.error(f"Error parsing EFetch XML: {str(e)}")
        return []

def extract_keywords_and_date(query, search_older=False, start_year=None):
    try:
        query_lower = query.lower()
        tokens = word_tokenize(query)
        tagged = pos_tag(tokens)
        stop_words = set(stopwords.words('english')).union({
            'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show', 
            'between', 'only', 'related', 'to', 'available'
        })
        
        keywords = []
        current_phrase = []
        for word, tag in tagged:
            if word.lower() in stop_words:
                if current_phrase:
                    keywords.append(' '.join(current_phrase))
                    current_phrase = []
                continue
            if tag.startswith('NN') or tag.startswith('JJ'):
                current_phrase.append(word.lower())
            else:
                if current_phrase:
                    keywords.append(' '.join(current_phrase))
                    current_phrase = []
        if current_phrase:
            keywords.append(' '.join(current_phrase))
        
        split_keywords = []
        for kw in keywords:
            if ' ' in kw:
                split_keywords.extend(kw.split())
            else:
                split_keywords.append(kw)
        
        keywords = list(set(split_keywords))[:5]
        
        keywords_with_synonyms = []
        for kw in keywords:
            synonyms = BIOMEDICAL_VOCAB.get(kw.lower(), [])[:2]
            keywords_with_synonyms.append((kw, synonyms))
        
        today = datetime.now()
        default_start_year = today.year - 10
        date_range = None
        
        if since_match := re.search(r'\bsince\s+(20\d{2})\b', query_lower):
            start_year = int(since_match.group(1))
            date_range = f"{start_year}/01/01:{today.strftime('%Y/%m/%d')}"
        elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
            year = int(year_match.group(1))
            date_range = f"{year}/01/01:{year}/12/31"
        elif 'past year' in query_lower:
            date_range = f"{(today - timedelta(days=365)).strftime('%Y/%m/%d')}:{today.strftime('%Y/%m/%d')}"
        elif 'past week' in query_lower:
            date_range = f"{(today - timedelta(days=7)).strftime('%Y/%m/%d')}:{today.strftime('%Y/%m/%d')}"
        else:
            start_year_int = int(start_year) if search_older and start_year else default_start_year
            date_range = f"{start_year_int}/01/01:{today.strftime('%Y/%m/%d')}"
        
        logger.info(f"Extracted keywords: {keywords_with_synonyms}, Date range: {date_range}")
        return keywords_with_synonyms, date_range, start_year_int if search_older and start_year else default_start_year
    except Exception as e:
        logger.error(f"Error extracting keywords and date: {str(e)}")
        return [], None, None

def build_pubmed_query(keywords_with_synonyms, date_range):
    try:
        query_parts = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
            term_query = " OR ".join(terms)
            query_parts.append(f"({term_query})")
        
        query = " AND ".join(query_parts) if query_parts else ""
        query = f"({query}) AND {date_range}[dp]" if query else f"{date_range}[dp]"
        logger.info(f"Built PubMed query: {query}")
        return query
    except Exception as e:
        logger.error(f"Error building PubMed query: {str(e)}")
        return ""

class SearchHandler:
    def __init__(self):
        self.source_id = "generic"
        self.name = "Generic Search"
    
    def search(self, query, keywords_with_synonyms, date_range, start_year):
        return [], []
    
    def query_grok_api(self, system_prompt, context):
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000
            )
            response = completion.choices[0].message.content.strip()
            response = re.sub(r'[^\x20-\x7E\n]', '', response)
            return response
        except Exception as e:
            logger.error(f"Error querying Grok API: {str(e)}")
            raise

class PubMedSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "pubmed"
        self.name = "PubMed"
    
    def search(self, query, keywords_with_synonyms, date_range, start_year):
        try:
            pubmed_query = build_pubmed_query(keywords_with_synonyms, date_range)
            logger.info(f"Executing PubMed search: {pubmed_query}")
            api_key = os.environ.get('PUBMED_API_KEY')
            pmids = esearch(pubmed_query, retmax=100, date_range=date_range, start_year=start_year, api_key=api_key)
            logger.info(f"PubMed ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                today = datetime.now()
                fallback_date_range = f"{today.year-10}/01/01:{today.strftime('%Y/%m/%d')}"
                fallback_query = build_pubmed_query(keywords_with_synonyms, fallback_date_range)
                logger.info(f"Executing PubMed fallback search: {fallback_query}")
                pmids = esearch(fallback_query, retmax=100, date_range=fallback_date_range, api_key=api_key)
                logger.info(f"PubMed fallback ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                return [], []
            
            efetch_xml = efetch(pmids, api_key=api_key)
            results = parse_efetch_xml(efetch_xml)
            
            primary_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) >= start_year
            ]
            fallback_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) < start_year
            ]
            
            logger.info(f"PubMed results: {len(primary_results)} primary, {len(fallback_results)} fallback")
            return primary_results, fallback_results
        except Exception as e:
            logger.error(f"Error in PubMed search: {str(e)}")
            return [], []

class GoogleScholarSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "googlescholar"
        self.name = "Google Scholar"
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=5, max=30),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(f"Retrying Google Scholar search, attempt {retry_state.attempt_number}")
    )
    def search(self, query, keywords_with_synonyms, date_range, start_year):
        try:
            keywords = " ".join([kw for kw, _ in keywords_with_synonyms])
            api_key = os.environ.get('SCRAPERAPI_KEY')
            if not api_key:
                logger.warning("SCRAPERAPI_KEY not set")
                return [], []
            
            url = f"https://api.scraperapi.com?api_key={api_key}&url=https://scholar.google.com/scholar?q={keywords}&num=10"
            response = requests.get(url, timeout=30, verify=False)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            for item in soup.find_all('div', class_='gs_r gs_or gs_scl'):
                title_elem = item.find('h3', class_='gs_rt')
                title = title_elem.text.strip() if title_elem else 'No title'
                
                link_elem = title_elem.find('a') if title_elem else None
                url = link_elem['href'] if link_elem and link_elem.get('href') else 'N/A'
                
                abstract_elem = item.find('div', class_='gs_rs')
                abstract = abstract_elem.text.strip() if abstract_elem else ''
                
                authors_elem = item.find('div', class_='gs_a')
                authors = authors_elem.text.strip() if authors_elem else 'N/A'
                
                if title == 'No title' and not abstract:
                    continue
                
                results.append({
                    'title': title,
                    'abstract': abstract,
                    'authors': authors,
                    'journal': 'Google Scholar',
                    'publication_date': 'N/A',
                    'url': url
                })
            
            logger.info(f"Google Scholar returned {len(results)} results for query: {keywords}")
            return results, []
        except Exception as e:
            logger.error(f"Error in Google Scholar search: {str(e)}")
            return [], []