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

logger = logging.getLogger(__name__)

BIOMEDICAL_VOCAB = {
    "diabetes": ["Diabetes Mellitus", "insulin resistance"],
    "weight loss": ["obesity", "body weight reduction"],
    "treatment": ["therapy", "intervention"],
    "disease": ["disorder", "condition"],
    "statins": ["HMG-CoA reductase inhibitors", "atorvastatin"],
    "heart disease": ["cardiovascular disease", "coronary artery disease"],
    "cardiovascular": ["heart-related", "circulatory"],
    "blood pressure": ["hypertension", "BP"],
    "hypertension": ["high blood pressure", "elevated BP"],
    "risk": ["hazard", "danger"],
    "smoking": ["smoke", "tobacco"],
    "pregnancy": ["gestation", "maternity"]
}

def esearch(term, db='pubmed', retmax=100, date_range=None, start_year=None):
    base_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={db}&term={term}&retmax={retmax}&retmode=json"
    if date_range:
        base_url += f"&datetype=pdat&mindate={date_range.split(':')[0]}&maxdate={date_range.split(':')[1]}"
    if start_year:
        base_url += f"&datetype=pdat&mindate={start_year}/01/01"
    
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('esummaryresult', {}).get('idlist', [])
    except Exception as e:
        logger.error(f"Error in ESearch: {str(e)}")
        return []

def extract_keywords_and_date(query, search_older=False, start_year=None):
    try:
        query_lower = query.lower()
        tokens = word_tokenize(query)
        tagged = pos_tag(tokens)
        stop_words = set(stopwords.words('english')).union({
            'provide', 'information', 'discuss', 'discusses', 'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 
            'of', 'for', 'any', 'articles', 'that', 'show', 'between', 'only', 'related', 'to', 'available', 'looking'
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
        default_start_year = today.year - 5
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
            pmids = esearch(pubmed_query, retmax=100, date_range=date_range, start_year=start_year)
            logger.info(f"PubMed ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                # Fallback query with broader date range (10 years)
                today = datetime.now()
                fallback_date_range = f"{today.year-10}/01/01:{today.strftime('%Y/%m/%d')}"
                fallback_query = " AND ".join([f'"{kw}"[All Fields]' for kw, _ in keywords_with_synonyms]) if keywords_with_synonyms else query
                fallback_query = f"({fallback_query}) AND {fallback_date_range}[dp]"
                logger.info(f"Executing PubMed fallback search: {fallback_query}")
                pmids = esearch(fallback_query, retmax=100, date_range=fallback_date_range)
                logger.info(f"PubMed fallback ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                return [], []
            
            efetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={','.join(pmids)}&retmode=xml"
            response = requests.get(efetch_url, timeout=20)
            response.raise_for_status()
            xml_data = ET.fromstring(response.content)
            
            results = []
            for article in xml_data.findall('.//PubmedArticle'):
                pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else 'N/A'
                title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else 'No title'
                
                abstract_elem = article.find('.//Abstract/AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ''
                
                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('LastName').text if author.find('LastName') is not None else ''
                    initials = author.find('Initials').text if author.find('Initials') is not None else ''
                    if last_name:
                        authors.append(f"{last_name} {initials}")
                authors_str = ", ".join(authors) if authors else "N/A"
                
                journal = article.find('.//Journal/Title').text if article.find('.//Journal/Title') is not None else 'N/A'
                
                pub_date = article.find('.//PubDate')
                if pub_date is not None:
                    year = pub_date.find('Year').text if pub_date.find('Year') is not None else ''
                    month = pub_date.find('Month').text if pub_date.find('Month') is not None else ''
                    day = pub_date.find('Day').text if pub_date.find('Day') is not None else ''
                    date_str = f"{year}-{month}-{day}" if year and month and day else year
                else:
                    date_str = 'N/A'
                
                result = {
                    'title': title,
                    'abstract': abstract,
                    'authors': authors_str,
                    'journal': journal,
                    'publication_date': date_str,
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
                logger.info(f"Parsed article: PMID={pmid}, Date={date_str}, Abstract={'Present' if abstract else 'Absent'}")
                results.append(result)
            
            logger.info(f"PubMed results: {len(results)} primary, 0 fallback")
            return results, []
        except Exception as e:
            logger.error(f"Error in PubMed search: {str(e)}")
            return [], []

class GoogleScholarSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "googlescholar"
        self.name = "Google Scholar"
    
    def search(self, query, keywords_with_synonyms, date_range, start_year):
        try:
            keywords = " ".join([kw for kw, _ in keywords_with_synonyms])
            api_key = os.environ.get('SCRAPERAPI_KEY')
            if not api_key:
                logger.warning("SCRAPERAPI_KEY not set, using fallback search")
                return [], []
            
            url = f"https://api.scraperapi.com?api_key={api_key}&url=https://scholar.google.com/scholar?q={keywords}&num=20"
            response = requests.get(url, timeout=20, verify=False)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
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