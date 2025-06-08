import requests
import xml.etree.ElementTree as ET
import logging
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json
from openai import OpenAI
import os
import re
from datetime import datetime

logger = logging.getLogger(__name__)

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
        tokens = word_tokenize(query.lower())
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        keywords = []
        for token in tokens:
            if token.isalnum() and token not in stop_words:
                lemma = lemmatizer.lemmatize(token)
                synonyms = [lemma]
                for syn in wordnet.synsets(lemma):
                    for l in syn.lemmas():
                        if l.name().lower() not in synonyms:
                            synonyms.append(l.name().lower())
                keywords.append((lemma, synonyms[:3]))
        
        today = datetime.now()
        if search_older:
            date_range = f"{start_year or 1900}/01/01:{today.strftime('%Y/%m/%d')}"
        else:
            date_range = f"{today.strftime('%Y')}/01/01:{today.strftime('%Y/%m/%d')}"
        
        start_year_int = int(start_year) if start_year else None
        return keywords, date_range, start_year_int
    except Exception as e:
        logger.error(f"Error extracting keywords and date: {str(e)}")
        return [], None, None

def build_pubmed_query(keywords, date_range):
    try:
        query_parts = []
        for keyword, synonyms in keywords:
            syn_query = " OR ".join([f'"{syn}"[All Fields]' for syn in [keyword] + synonyms])
            query_parts.append(f"({syn_query})")
        query = " AND ".join(query_parts)
        query += f" AND {date_range}[dp]"
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
    
    def rank_results(self, query, results, prompt_params):
        return results
    
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
            return completion.choices[0].message.content
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
    
    def rank_results(self, query, results, prompt_params):
        try:
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}" for r in results])
            system_prompt = f"""
            Rank the following articles by relevance to the query '{query}' (1 = most relevant). Focus on how well each article addresses smoking and pregnancy. Provide a JSON array of objects with 'index' (original position, 0-based) and 'explanation' (brief reason for ranking). Limit to top {len(results)} results.
            """
            response = self.query_grok_api(system_prompt, context)
            ranked_indices = json.loads(response)
            logger.info(f"Grok ranking response for pubmed: {response[:200]}...")
            
            # Validate and sort results
            ranked_results = []
            seen_indices = set()
            for rank in ranked_indices:
                idx = rank.get('index', -1)
                if 0 <= idx < len(results) and idx not in seen_indices:
                    ranked_results.append(results[idx])
                    seen_indices.add(idx)
            
            # Add unranked results
            for i, result in enumerate(results):
                if i not in seen_indices:
                    ranked_results.append(result)
            
            return ranked_results
        except Exception as e:
            logger.error(f"Failed to rank PubMed results: {str(e)}")
            return results  # Fallback to original order

class FDASearchHandler(SearchHandler):
    def __init__(self):
        super().__init__()
        self.source_id = "fda"
        self.name = "FDA.gov"
    
    def search(self, query, keywords_with_synonyms, date_range, start_year):
        try:
            keywords = " ".join([syn for kw, syns in keywords_with_synonyms for syn in ([kw] + syns)])
            url = f"https://api.fda.gov/drug/label.json?search={keywords}&limit=100"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                title = item.get('openfda', {}).get('brand_name', ['Unknown'])[0]
                abstract = item.get('description', [''])[0] or item.get('indications_and_usage', [''])[0]
                date = item.get('effective_time', 'N/A')
                # Sanitize abstract to prevent JSON parsing issues
                abstract = abstract.replace('"', "'").replace('\n', ' ').strip()
                if title and abstract:
                    results.append({
                        'title': title,
                        'abstract': abstract[:1000],  # Limit length
                        'url': f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={item.get('id', '')}",
                        'publication_date': date,
                        'authors': 'N/A',
                        'journal': 'FDA.gov'
                    })
            
            logger.info(f"FDA API returned {len(results)} results for query: {keywords}")
            return results, []
        except Exception as e:
            logger.error(f"Error in FDA search: {str(e)}")
            return [], []
    
    def rank_results(self, query, results, prompt_params):
        try:
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nDate: {r.get('publication_date', 'N/A')}" for r in results])
            system_prompt = f"""
            Rank the following FDA articles by relevance to the query '{query}' (1 = most relevant). Focus on smoking cessation or pregnancy-related content. Provide a JSON array of objects with 'index' (original position, 0-based) and 'explanation' (brief reason for ranking). Limit to top {len(results)} results.
            """
            response = self.query_grok_api(system_prompt, context)
            ranked_indices = json.loads(response)
            logger.info(f"Grok ranking response for fda: {response[:200]}...")
            
            # Validate and sort results
            ranked_results = []
            seen_indices = set()
            for rank in ranked_indices:
                idx = rank.get('index', -1)
                if 0 <= idx < len(results) and idx not in seen_indices:
                    ranked_results.append(results[idx])
                    seen_indices.add(idx)
            
            # Add unranked results
            for i, result in enumerate(results):
                if i not in seen_indices:
                    ranked_results.append(result)
            
            return ranked_results
        except Exception as e:
            logger.error(f"Failed to rank FDA results: {str(e)}")
            return results  # Fallback to original order

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
                raise ValueError("SCRAPERAPI_KEY not set")
            
            url = f"https://api.scraperapi.com?api_key={api_key}&url=https://scholar.google.com/scholar?q={keywords}&num=10"
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
    
    def rank_results(self, query, results, prompt_params):
        try:
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}" for r in results])
            system_prompt = f"""
            Rank the following articles by relevance to the query '{query}' (1 = most relevant). Provide a JSON array of objects with 'index' (original position, 0-based) and 'explanation' (brief reason for ranking). Limit to top {len(results)} results.
            """
            response = self.query_grok_api(system_prompt, context)
            ranked_indices = json.loads(response)
            logger.info(f"Grok ranking response for googlescholar: {response[:200]}...")
            
            ranked_results = []
            seen_indices = set()
            for rank in ranked_indices:
                idx = rank.get('index', -1)
                if 0 <= idx < len(results) and idx not in seen_indices:
                    ranked_results.append(results[idx])
                    seen_indices.add(idx)
            
            for i, result in enumerate(results):
                if i not in seen_indices:
                    ranked_results.append(result)
            
            return ranked_results
        except Exception as e:
            logger.error(f"Failed to rank Google Scholar results: {str(e)}")
            return results