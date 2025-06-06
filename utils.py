import requests
import logging
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from urllib.parse import quote
import json
from abc import ABC, abstractmethod
from openai import OpenAI
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from bs4 import BeautifulSoup
import time
import hashlib
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

# Biomedical vocabulary for synonyms
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
    "cidp": ["Chronic Inflammatory Demyelinating Polyneuropathy", "demyelinating polyneuropathy"]
}

class SearchHandler(ABC):
    def __init__(self, source_id, name, max_results=80, summary_limit=20):
        self.source_id = source_id
        self.name = name
        self.max_results = max_results
        self.summary_limit = summary_limit
        self.api_key = None

    @abstractmethod
    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        pass

    @abstractmethod
    def format_result(self, result):
        pass

    def rank_results(self, query, results, prompt_params):
        display_result_count = prompt_params.get('display_result_count', self.max_results)
        sort_by = prompt_params.get('sort_by', 'relevance')
        if not results:
            return []
        
        if sort_by == 'date':
            sorted_results = sorted(results, key=lambda x: x.get('publication_date', x.get('date', '0')), reverse=True)
            return sorted_results[:display_result_count]
        
        try:
            max_rank_results = min(len(results), 50)
            articles_context = []
            for i, result in enumerate(results[:max_rank_results]):
                article_text = f"Article {i+1}: {self._format_context(result)}"
                articles_context.append(article_text)
            
            context = "\n\n".join(articles_context)
            ranking_prompt = f"""
Given the query '{query}', rank the following articles by relevance.
Focus on articles that directly address the query's topic and intent.
Exclude articles that are unrelated to the query.
Return a JSON list of article indices (1-based) in order of relevance, with a brief explanation for each.
Ensure the response is valid JSON. Example:
[
    {{"index": 1, "explanation": "Directly discusses the query topic with high relevance"}},
    {{"index": 2, "explanation": "Relevant but less specific to the query"}}
]
Articles:
{context}
"""
            response = self._query_grok_api(query, context, ranking_prompt)
            logger.info(f"Grok ranking response for {self.source_id}: {response[:200]}...")
            
            try:
                ranking = json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Grok ranking response for {self.source_id}: {str(e)}, response: {response[:500]}...")
                return results[:display_result_count]
            
            if isinstance(ranking, dict) and 'articles' in ranking:
                ranking = ranking['articles']
            if not isinstance(ranking, list):
                logger.warning(f"Invalid Grok ranking response format for {self.source_id}: {response[:200]}...")
                return results[:display_result_count]
            
            ranked_indices = []
            for item in ranking:
                if isinstance(item, dict) and 'index' in item:
                    index = int(item['index']) - 1
                    if 0 <= index < max_rank_results:
                        ranked_indices.append(index)
            
            missing_indices = [i for i in range(len(results)) if i not in ranked_indices]
            ranked_indices.extend(missing_indices)
            
            ranked_results = [results[i] for i in ranked_indices[:display_result_count]]
            logger.info(f"{self.source_id} ranked {len(ranked_results)} results: indices {ranked_indices[:display_result_count]}")
            return ranked_results
        except Exception as e:
            logger.error(f"{self.source_id} ranking failed: {str(e)}")
            return results[:display_result_count]

    def generate_summary(self, query, results, prompt_text, prompt_params):
        if not results:
            return f"No results found for '{query}' in {self.name}."
        
        context_results = results[:self.summary_limit]
        context = "\n".join([self._format_context(r) for r in context_results])
        
        MAX_CONTEXT_LENGTH = 12000
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
            logger.warning(f"Context truncated to {MAX_CONTEXT_LENGTH} characters for {self.source_id} query: {query[:50]}...")
        
        try:
            output = self._query_grok_api(query, context, prompt_text)
            paragraphs = output.split('\n\n')
            formatted_output = ''.join(f'<p>{p}</p>' for p in paragraphs if p.strip())
            logger.info(f"Generated {self.source_id} summary: length={len(formatted_output)}")
            return formatted_output
        except Exception as e:
            logger.error(f"Error generating {self.source_id} summary: {str(e)}")
            return f"Unable to generate AI summary for {self.name}: {str(e)}."

    def _query_grok_api(self, query, context, prompt):
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            completion = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Based on the following context, answer the prompt: {query}\n\nContext: {context}"}
                ],
                max_tokens=1000
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying Grok API for {self.source_id}: {str(e)}")
            raise

    def _format_context(self, result):
        return f"Title: {result['title']}\nAbstract: {result.get('abstract', result.get('summary', ''))}\nAuthors: {result.get('authors', 'N/A')}\nDate: {result.get('publication_date', result.get('date', 'N/A'))}"

class PubMedSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="pubmed", name="PubMed")
        self.api_key = os.environ.get('PUBMED_API_KEY')

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        search_query = build_pubmed_query(keywords_with_synonyms, date_range)
        logger.info(f"Executing PubMed search: {search_query}")
        esearch_result = esearch(search_query, retmax=self.max_results, api_key=self.api_key)
        pmids = esearch_result.get('esearchresult', {}).get('idlist', [])
        logger.info(f"PubMed ESearch result: {len(pmids)} PMIDs")
        
        results = []
        if pmids:
            efetch_xml = efetch(pmids, api_key=self.api_key)
            results = parse_e_fetch_xml(efetch_xml) or []
        
        primary_results = [
            r for r in results
            if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) >= start_year_int
        ]
        fallback_results = [
            r for r in results
            if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) < start_year_int
        ]
        
        logger.info(f"PubMed results: {len(primary_results)} primary, {len(fallback_results)} fallback")
        return primary_results, fallback_results

    def format_result(self, result):
        return {
            "id": result["id"],
            "title": result["title"],
            "abstract": result["abstract"],
            "authors": result["authors"],
            "journal": result["journal"],
            "publication_date": result["publication_date"],
            "url": result["url"]
        }

class FDASearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="fda", name="FDA.gov")

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        results = search_fda_label_api(query, keywords_with_synonyms, date_range)
        logger.info(f"FDA results: {len(results)}")
        return results, []

    def format_result(self, result):
        return {
            "id": result["id"],
            "title": result["title"],
            "summary": result["summary"],
            "date": result["date"],
            "url": result["url"]
        }

logger = logging.getLogger(__name__)

class GoogleScholarSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="googlescholar", name="Google Scholar")

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        search_terms = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [keyword] + synonyms
            search_terms.extend([t.strip().replace('+', ' ') for t in terms if t.strip()])
        if not search_terms:
            return [], []
        
        search_query = ' '.join(search_terms)
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429], allowed_methods=["GET"])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        try:
            base_url = "https://scholar.google.com/scholar"
            params = {
                "q": search_query,
                "hl": "en",
                "as_ylo": start_year_int,
                "as_yhi": datetime.now().year
            }
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            response = session.get(base_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            for entry in soup.select('.gs_ri')[:self.max_results]:
                title_elem = entry.select_one('.gs_rt a')
                title = title_elem.text if title_elem else "No title"
                url = title_elem['href'] if title_elem else ""
                abstract_elem = entry.select_one('.gs_rs')
                abstract = abstract_elem.text if abstract_elem else ""
                authors_elem = entry.select_one('.gs_a')
                authors = authors_elem.text.split(' - ')[0] if authors_elem else "N/A"
                year = ""
                if authors_elem:
                    year_match = re.search(r'\b(\d{4})\b', authors_elem.text)
                    year = year_match.group(1) if year_match else "N/A"
                
                results.append({
                    "id": hashlib.md5(url.encode()).hexdigest(),
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "journal": "N/A",
                    "publication_date": year,
                    "url": url
                })
            
            logger.info(f"Google Scholar returned {len(results)} results for query: {search_query}")
            primary_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) >= start_year_int
            ]
            fallback_results = [
                r for r in results
                if r['publication_date'] and r['publication_date'].isdigit() and int(r['publication_date']) < start_year_int
            ]
            return primary_results, fallback_results
        except Exception as e:
            logger.error(f"Error querying Google Scholar: {str(e)}")
            return [], []

    def format_result(self, result):
        return {
            "id": result["id"],
            "title": result["title"],
            "abstract": result["abstract"],
            "authors": result["authors"],
            "journal": result["journal"],
            "publication_date": result["publication_date"],
            "url": result["url"]
        }

def esearch(query, retmax=100, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": str(query),
        "retmax": retmax,
        "retmode": "json",
        "usehistory": "y"
    }
    if api_key:
        params["api_key"] = api_key
    try:
        logger.info(f"PubMed ESearch query: {query}")
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error in ESearch: {str(e)}")
        return {"esearchresult": {"idlist": []}}

def efetch(pmids, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(map(str, pmids)),
        "retmode": "xml",
        "rettype": "abstract"
    }
    if api_key:
        params["api_key"] = api_key
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error in EFetch: {str(e)}")
        return ""

def parse_e_fetch_xml(xml_data):
    try:
        root = ET.fromstring(xml_data)
        articles = []
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else ""
            title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else "No title"
            abstract_elem = article.find(".//Abstract/AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            authors = []
            for author in article.findall(".//Author"):
                lastname = author.find("LastName")
                initials = author.find("Initials")
                if lastname is not None and initials is not None:
                    authors.append(f"{lastname.text} {initials.text}")
            authors_str = ", ".join(authors) if authors else "N/A"
            journal = article.find(".//Journal/Title").text if article.find(".//Journal/Title") is not None else "N/A"
            pub_date = article.find(".//PubDate/Year")
            publication_date = pub_date.text if pub_date is not None else "N/A"
            logger.info(f"Parsed article: PMID={pmid}, Date={publication_date}, Abstract={'Present' if abstract else 'Absent'}")
            articles.append({
                "id": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "journal": journal,
                "publication_date": publication_date,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
            })
        return articles
    except Exception as e:
        logger.error(f"Error parsing EFetch XML: {str(e)}")
        return []

def search_fda_label_api(query, keywords_with_synonyms, date_range):
    try:
        search_terms = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [keyword] + synonyms
            search_terms.extend([t.strip().replace('+', ' ') for t in terms if t.strip()])
        if not search_terms:
            return []
        search_query = quote(' '.join(search_terms))
        
        date_filter = ""
        if date_range:
            start_date = date_range.split(':')[0].replace('/', '')[:8]
            end_date = date_range.split(':')[1].replace('[dp]', '').replace('/', '')[:8]
            date_filter = f"+AND+effective_time:[{start_date}+TO+{end_date}]"
        
        url = f"https://api.fda.gov/drug/label.json?search={search_query}{date_filter}&limit=20"
        headers = {'User-Agent': 'PubMedResearch/1.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('results', []):
            title = item.get('openfda', {}).get('brand_name', ['No title'])[0] or 'No title'
            summary = item.get('indications_and_usage', ['No summary'])[0] or 'No summary'
            date = item.get('effective_time', '')[:8] or 'N/A'
            results.append({
                'id': item.get('id', ''),
                'title': title,
                'summary': summary,
                'date': date,
                'url': f"https://open.fda.gov/data/drug/label/{item.get('id', '')}"
            })
        logger.info(f"FDA API returned {len(results)} results for query: {search_query}")
        return results
    except Exception as e:
        logger.error(f"Error querying FDA API: {str(e)}")
        return []

def extract_keywords_and_date(query, search_older=False, start_year=None):
    query_lower = query.lower()
    tokens = word_tokenize(query)
    tagged = pos_tag(tokens)
    stop_words = set(stopwords.words('english')).union({
        'what', 'can', 'you', 'tell', 'me', 'is', 'are', 'how', 'why', 'when', 'where', 'who',
        'which', 'please', 'about', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'related',
        'any', 'articles', 'that', 'show', 'between', 'only', 'latest', 'recent', 'new'
    })
    
    keywords = []
    current_phrase = []
    for word, tag in tagged:
        if word.lower() in stop_words:
            if current_phrase:
                keywords.append('+'.join(current_phrase))
                current_phrase = []
            continue
        if tag.startswith('NN') or tag.startswith('JJ'):
            current_phrase.append(word.lower())
        else:
            if current_phrase:
                keywords.append('+'.join(current_phrase))
                current_phrase = []
    if current_phrase:
        keywords.append('+'.join(current_phrase))
    
    split_keywords = []
    for kw in keywords:
        if '+' in kw:
            split_keywords.extend(kw.split('+'))
        split_keywords.append(kw)
    
    keywords = list(set(split_keywords))[:5]
    
    keywords_with_synonyms = []
    for kw in keywords:
        synonyms = BIOMEDICAL_VOCAB.get(kw.lower(), [])[:2]
        keywords_with_synonyms.append((kw, synonyms))
    
    today = datetime.now()
    default_start_year = today.year - 5
    date_range = None
    start_year_int = default_start_year
    
    timeframe_patterns = [
        (r'within\s+the\s+last\s+(\d+)\s+year(s?)', lambda m: int(m.group(1))),
        (r'in\s+(\d{4})', lambda m: (int(m.group(1)), int(m.group(1)))),
        (r'since\s+(\d{4})', lambda m: (int(m.group(1)), today.year)),
        (r'from\s+(\d{4})\s+to\s+(\d{4})', lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'from\s+(\d{4})\s+and\s+(\d{4})', lambda m: (int(m.group(1)), int(m.group(2)))),
        (r'past\s+year', lambda m: 1),
        (r'past\s+week', lambda m: 0.019178)
    ]
    
    for pattern, extractor in timeframe_patterns:
        if match := re.search(pattern, query_lower):
            result = extractor(match)
            if isinstance(result, tuple):
                start_year, end_year = result
            else:
                start_year = today.year - result
                end_year = today.year
            start_date = f"{start_year}/01/01"
            end_date = f"{end_year}/12/31" if isinstance(result, tuple) else today.strftime('%Y/%m/%d')
            date_range = f"{start_date}[dp]:{end_date}[dp]"
            start_year_int = start_year
            break
    
    if not date_range and not search_older:
        start_date = (today - timedelta(days=5*365)).strftime('%Y/%m/%d')
        end_date = today.strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_date[:4])
    elif search_older and start_year:
        start_date = f"{start_year}/01/01"
        end_date = today.strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_year)
    
    logger.info(f"Extracted keywords: {keywords_with_synonyms}, Date range: {date_range}")
    return keywords_with_synonyms, date_range, start_year_int

def build_pubmed_query(keywords_with_synonyms, date_range):
    if not keywords_with_synonyms:
        return ""
    terms = []
    for keyword, synonyms in keywords_with_synonyms:
        formatted_keyword = keyword.replace('+', ' ')
        all_terms = [f'"{formatted_keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        terms.append("(" + " OR ".join(all_terms) + ")")
    query = " AND ".join(terms)
    if date_range:
        query += f" AND {date_range}"
    logger.info(f"Built PubMed query: {query}")
    return query