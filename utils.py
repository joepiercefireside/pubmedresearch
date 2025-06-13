import requests
import xml.etree.ElementTree as ET
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
import os
import re
from datetime import datetime, timedelta
import tenacity
from ratelimit import limits, sleep_and_retry
from bs4 import BeautifulSoup
import urllib.parse
import random
from search_utils import rank_results
import time

logger = logging.getLogger(__name__)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
]

BIOMEDICAL_VOCAB = {
    "diabetes": ["Diabetes Mellitus", "insulin resistance", "type 2 diabetes"],
    "weight loss": ["obesity", "body weight reduction", "fat loss"],
    "treatment": ["therapy", "intervention", "management"],
    "disease": ["disorder", "condition", "pathology"],
    "statins": ["HMG-CoA reductase inhibitors", "atorvastatin", "simvastatin"],
    "heart disease": ["cardiovascular disease", "coronary artery disease", "myocardial infarction"],
    "cardiovascular": ["heart-related", "circulatory"],
    "blood pressure": ["hypertension", "BP"],
    "hypertension": ["high blood pressure", "elevated BP"],
    "cidp": ["chronic inflammatory demyelinating polyradiculoneuropathy"],
    "adverse event": ["side effect", "adverse reaction", "untoward effect"],
    "adverse effect": ["side effect", "adverse reaction", "negative effect"]
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
            abstract = abstract_elem.text or '' if abstract_elem is not None else ""
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
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "source_id": "pubmed"
            })
        return articles
    except Exception as e:
        logger.error(f"Error parsing EFetch XML: {str(e)}")
        return []

def get_mesh_synonyms(keyword, api_key=None):
    if keyword.lower() in BIOMEDICAL_VOCAB:
        logger.info(f"Using BIOMEDICAL_VOCAB for {keyword}")
        return BIOMEDICAL_VOCAB[keyword.lower()][:3]
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=mesh&term={urllib.parse.quote(keyword)}&retmax=1&retmode=xml"
    if api_key:
        url += f"&api_key={api_key}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        logger.debug(f"MeSH API response for {keyword}: {response.content[:200]}...")
        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logger.error(f"Invalid XML in MeSH response for {keyword}: {str(e)}")
            return BIOMEDICAL_VOCAB.get(keyword.lower(), get_datamuse_synonyms(keyword))
        id_list = root.findall(".//Id")
        if not id_list:
            logger.info(f"No MeSH descriptors found for {keyword}")
            return get_datamuse_synonyms(keyword)
        
        descriptor_id = id_list[0].text
        efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=mesh&id={descriptor_id}&retmode=xml"
        if api_key:
            efetch_url += f"&api_key={api_key}"
        efetch_response = requests.get(efetch_url, timeout=5)
        efetch_response.raise_for_status()
        try:
            efetch_root = ET.fromstring(efetch_response.content)
        except ET.ParseError as e:
            logger.error(f"Invalid XML in MeSH efetch response for {keyword}: {str(e)}")
            return BIOMEDICAL_VOCAB.get(keyword.lower(), get_datamuse_synonyms(keyword))
        
        synonyms = []
        for term in efetch_root.findall(".//TermList/Term"):
            if term.text:
                synonyms.append(term.text.lower())
        for entry_term in efetch_root.findall(".//ConceptList/Concept/TermList/Term"):
            if entry_term.text:
                synonyms.append(entry_term.text.lower())
        
        return list(set(synonyms))[:3]
    except Exception as e:
        logger.error(f"MeSH API error for {keyword}: {str(e)}")
        return BIOMEDICAL_VOCAB.get(keyword.lower(), get_datamuse_synonyms(keyword))

def get_datamuse_synonyms(keyword):
    url = f"https://api.datamuse.com/words?rel_syn={urllib.parse.quote(keyword)}&max=5"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return [item['word'].lower() for item in data if 'word' in item][:3]
    except Exception as e:
        logger.error(f"Datamuse API error for {keyword}: {str(e)}")
        return []

def extract_keywords_and_date(query, search_older=False, start_year=None):
    try:
        query_lower = query.lower()
        tokens = word_tokenize(query)
        tagged = pos_tag(tokens)
        stop_words = set(stopwords.words('english')).union({
            'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show',
            'between', 'only', 'related', 'to', 'available', 'discuss', 'provide', 'and'
        })
        
        keywords = []
        current_phrase = []
        for i, (word, tag) in enumerate(tagged):
            if word.lower() in stop_words:
                if current_phrase:
                    keywords.append(' '.join(current_phrase))
                    current_phrase = []
                continue
            if tag.startswith('NN') or tag.startswith('JJ'):
                current_phrase.append(word.lower())
            elif current_phrase and (i == len(tagged)-1 or tagged[i+1][0].lower() in stop_words):
                keywords.append(' '.join(current_phrase))
                current_phrase = []
            else:
                if current_phrase:
                    keywords.append(' '.join(current_phrase))
                    current_phrase = []
        if current_phrase:
            keywords.append(' '.join(current_phrase))
        
        keywords = [kw for kw in keywords if kw.strip()][:5]
        
        api_key = os.environ.get('PUBMED_API_KEY')
        keywords_with_synonyms = []
        for kw in keywords:
            synonyms = get_mesh_synonyms(kw, api_key)
            if not synonyms and kw.lower() not in BIOMEDICAL_VOCAB:
                synonyms.extend(get_datamuse_synonyms(kw))
            keywords_with_synonyms.append((kw, list(set(synonyms))[:3]))
        
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
        or_group = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
            term_query = " OR ".join(terms)
            or_group.append(term_query)
        
        # Combine terms with AND, but group OR terms together
        if or_group:
            query = f"({' OR '.join(or_group)})"
        else:
            query = ""
        
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
    
    def search(self, query, keywords_with_synonyms, date_range, start_year, result_limit=50):
        return [], []

class PubMedSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "pubmed"
        self.name = "PubMed"
    
    def search(self, query, keywords_with_synonyms, date_range, start_year, result_limit=50):
        try:
            pubmed_query = build_pubmed_query(keywords_with_synonyms, date_range)
            logger.info(f"Executing PubMed search: {pubmed_query}")
            api_key = os.environ.get('PUBMED_API_KEY')
            pmids = esearch(pubmed_query, retmax=min(result_limit, 100), date_range=date_range, start_year=start_year, api_key=api_key)
            logger.info(f"PubMed ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                today = datetime.now()
                fallback_date_range = f"{today.year-10}/01/01:{today.strftime('%Y/%m/%d')}"
                fallback_query = build_pubmed_query(keywords_with_synonyms, fallback_date_range)
                logger.info(f"Executing PubMed fallback search: {fallback_query}")
                pmids = esearch(fallback_query, retmax=min(result_limit, 100), date_range=fallback_date_range, api_key=api_key)
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
            return primary_results[:result_limit], fallback_results[:result_limit]
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
        wait=tenacity.wait_exponential(multiplier=2, min=5, max=60),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(f"Retrying Google Scholar search, attempt {retry_state.attempt_number}")
    )
    def search(self, query, keywords_with_synonyms, date_range, start_year, result_limit=50):
        try:
            keywords = "+".join([kw.replace(" ", "+") for kw, _ in keywords_with_synonyms])
            serpapi_key = os.environ.get('SERPAPI_KEY')
            if not serpapi_key:
                logger.error("SERPAPI_KEY not set")
                raise ValueError("SERPAPI_KEY not set")
            
            logger.info(f"Using SerpApi for Google Scholar search: {keywords}")
            results = []
            results_per_page = 20
            pages_needed = (result_limit + results_per_page - 1) // results_per_page
            for page in range(min(pages_needed, 5)):  # Cap at 100 results (5 pages)
                start = page * results_per_page
                serpapi_url = f"https://serpapi.com/search?engine=google_scholar&q={keywords}&num={results_per_page}&start={start}&api_key={serpapi_key}"
                if start_year:
                    serpapi_url += f"&as_ylo={start_year}"
                
                response = requests.get(serpapi_url, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'organic_results' in data:
                    for item in data['organic_results']:
                        pub_date = item.get('publication_info', {}).get('summary', '').split(', ')[-1] if ', ' in item.get('publication_info', {}).get('summary', '') else 'N/A'
                        if pub_date != 'N/A' and start_year and pub_date.isdigit() and int(pub_date) < start_year:
                            continue
                        results.append({
                            'title': item.get('title', 'No title'),
                            'abstract': item.get('snippet', ''),
                            'authors': ', '.join([author.get('name', 'N/A') for author in item.get('publication_info', {}).get('authors', [])]) or 'N/A',
                            'journal': item.get('publication_info', {}).get('summary', 'Google Scholar').split(' - ')[0],
                            'publication_date': pub_date,
                            'url': item.get('link', ''),
                            'source_id': 'googlescholar'
                        })
                
                time.sleep(1)  # Avoid rate limits
            
            logger.info(f"SerpApi Google Scholar returned {len(results)} results for query: {keywords}")
            return results[:result_limit], []
        except Exception as e:
            logger.error(f"Error in Google Scholar search: {str(e)}")
            return [], []

class SemanticScholarSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "semanticscholar"
        self.name = "Semantic Scholar"
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=10, max=120),
        retry=tenacity.retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.info(f"Retrying Semantic Scholar search, attempt {retry_state.attempt_number}")
    )
    def search(self, query, keywords_with_synonyms, date_range, start_year, result_limit=50):
        try:
            keywords = " ".join([kw for kw, _ in keywords_with_synonyms])
            logger.info(f"Using Semantic Scholar API for search: {keywords}")
            ss_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(keywords)}&limit={min(result_limit, 100)}&fields=title,abstract,authors,journal,year,url"
            if start_year:
                ss_url += f"&year={start_year}-"
            
            time.sleep(2)  # Add delay to avoid rate limit
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            api_key = os.environ.get('SEMANTIC_SCHOLAR_API_KEY')
            if api_key:
                headers['x-api-key'] = api_key
            
            response = requests.get(ss_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            results = []
            if 'data' in data:
                for item in data['data']:
                    if start_year and item.get('year') and item['year'] < start_year:
                        continue
                    authors = [author.get('name', 'N/A') for author in item.get('authors', [])]
                    results.append({
                        'title': item.get('title', 'No title'),
                        'abstract': item.get('abstract', '') or '',
                        'authors': ', '.join(authors) if authors else 'N/A',
                        'journal': item.get('journal', {}).get('name', 'Semantic Scholar') or 'Semantic Scholar',
                        'publication_date': str(item.get('year', 'N/A')),
                        'url': item.get('url', ''),
                        'source_id': 'semanticscholar'
                    })
            
            logger.info(f"Semantic Scholar returned {len(results)} results for query: {keywords}")
            return results[:result_limit], []
        except Exception as e:
            logger.error(f"Error in Semantic Scholar search: {str(e)}")
            return [], []