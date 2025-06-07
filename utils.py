import requests
import xml.etree.ElementTree as ET
import logging
import json
import time
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import hashlib
import re
import os
from datetime import datetime
from openai import OpenAI
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag

logger = logging.getLogger(__name__)

class SearchHandler:
    def __init__(self, source_id, name):
        self.source_id = source_id
        self.name = name
        self.max_results = 80

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        raise NotImplementedError("Search method must be implemented by subclasses")

    def rank_results(self, query, results, prompt_params):
        return results

    def generate_summary(self, query, results, prompt_text, prompt_params):
        return ""

    def format_result(self, result):
        return result

class PubMedSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="pubmed", name="PubMed")

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        try:
            pubmed_query = build_pubmed_query(keywords_with_synonyms, date_range)
            logger.info(f"Executing PubMed search: {pubmed_query}")
            pmids = esearch(pubmed_query, max_results=self.max_results)
            logger.info(f"PubMed ESearch result: {len(pmids)} PMIDs")
            
            if not pmids:
                return [], []
            
            xml_data = efetch(pmids)
            results = parse_e_fetch_xml(xml_data)
            
            primary_results = []
            fallback_results = []
            
            for article in results:
                logger.info(f"Parsed article: PMID={article.get('pmid', 'N/A')}, Date={article.get('publication_date', 'N/A')}, Abstract={'Present' if article.get('abstract') else 'Absent'}")
                pub_year = article.get('publication_date', '')
                try:
                    year = int(pub_year.split('-')[0]) if pub_year else 0
                    if year >= start_year_int:
                        primary_results.append(self.format_result(article))
                    else:
                        fallback_results.append(self.format_result(article))
                except ValueError:
                    fallback_results.append(self.format_result(article))
            
            logger.info(f"PubMed results: {len(primary_results)} primary, {len(fallback_results)} fallback")
            return primary_results, fallback_results
        
        except Exception as e:
            logger.error(f"Error querying PubMed: {str(e)}")
            return [], []

    def rank_results(self, query, results, prompt_params):
        if not results:
            return []
        
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}" for r in results])
            prompt = f"""
            You are a research assistant tasked with ranking PubMed articles based on their relevance to the query: "{query}". 
            For each article, provide:
            - The index of the article in the provided list (0 to {len(results)-1}).
            - A brief explanation of why the article is relevant to the query, considering the title, abstract, authors, and publication date.
            Return the ranking as a JSON array of objects with "index" and "explanation" fields, ordered from most to least relevant.
            Limit the ranking to the top {prompt_params.get('display_result_count', 80)} articles.
            Ensure the response is valid JSON and contains only the array of ranking objects.
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model="grok-3",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": context}
                        ],
                        max_tokens=2000
                    )
                    
                    response_text = completion.choices[0].message.content.strip()
                    logger.info(f"Grok ranking response for pubmed: {response_text[:200]}...")
                    
                    # Remove code block markers if present
                    if response_text.startswith('```json'):
                        response_text = response_text[7:].rstrip('```')
                    elif response_text.startswith('```'):
                        response_text = response_text[3:].rstrip('```')
                    
                    # Attempt to parse JSON, truncate to last valid object if malformed
                    try:
                        ranked_indices = json.loads(response_text)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON parsing failed: {str(e)}, attempting to truncate response")
                        # Find last valid object
                        last_valid_pos = response_text.rfind('}')
                        if last_valid_pos > 0:
                            truncated_response = response_text[:last_valid_pos + 1] + ']'
                            try:
                                ranked_indices = json.loads(truncated_response)
                                logger.info(f"Successfully parsed truncated JSON response")
                            except json.JSONDecodeError as trunc_e:
                                logger.error(f"Failed to parse truncated response: {str(trunc_e)}")
                                if attempt < max_retries - 1:
                                    logger.info(f"Retrying API call, attempt {attempt + 2}")
                                    time.sleep(random.uniform(1, 3))
                                    continue
                                return results  # Fallback to unranked results
                        else:
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying API call, attempt {attempt + 2}")
                                time.sleep(random.uniform(1, 3))
                                continue
                            return results  # Fallback to unranked results
                    
                    valid_indices = {i for i in range(len(results))}
                    ranked_results = []
                    seen_indices = set()
                    
                    for item in ranked_indices:
                        index = item.get('index')
                        if index in valid_indices and index not in seen_indices:
                            ranked_results.append(results[index])
                            seen_indices.add(index)
                    
                    for i, result in enumerate(results):
                        if i not in seen_indices:
                            ranked_results.append(result)
                    
                    logger.info(f"pubmed ranked {len(ranked_results)} results: indices {[r.get('index', i) for i, r in enumerate(ranked_results)]}")
                    return ranked_results
                
                except Exception as api_e:
                    logger.error(f"API error on attempt {attempt + 1}: {str(api_e)}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying API call, attempt {attempt + 2}")
                        time.sleep(random.uniform(1, 3))
                        continue
                    logger.error(f"Max retries reached, returning unranked results")
                    return results
        
        except Exception as e:
            logger.error(f"Error ranking PubMed results: {str(e)}")
            return results

    def generate_summary(self, query, results, prompt_text, prompt_params):
        if not results or not prompt_text:
            return "No summary generated due to missing results or prompt."
        
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            context = "\n".join([f"Title: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}" for r in results[:prompt_params.get('summary_result_count', 20)]])
            
            completion = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000
            )
            
            summary = completion.choices[0].message.content.strip()
            logger.info(f"Generated pubmed summary: length={len(summary)}")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating PubMed summary: {str(e)}")
            return f"Unable to generate summary: {str(e)}"

    def format_result(self, result):
        return {
            "id": result.get("pmid", ""),
            "title": result.get("title", "No title"),
            "abstract": result.get("abstract", ""),
            "authors": result.get("authors", "N/A"),
            "journal": result.get("journal", "N/A"),
            "publication_date": result.get("publication_date", "N/A"),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{result.get('pmid', '')}"
        }

class FDASearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="fda", name="FDA.gov")

    def search(self, query, keywords_with_synonyms, date_range, start_year_int):
        try:
            search_terms = []
            for keyword, synonyms in keywords_with_synonyms:
                terms = [keyword] + synonyms
                search_terms.extend([t.strip().replace('+', ' ') for t in terms if t.strip()])
            
            if not search_terms:
                return [], []
            
            search_query = '%20'.join(search_terms)
            results = search_fda_label_api(search_query, limit=self.max_results)
            logger.info(f"FDA API returned {len(results)} results for query: {search_query}")
            
            primary_results = []
            fallback_results = []
            
            for result in results:
                pub_date = result.get('publication_date', '')
                try:
                    year = int(pub_date.split('-')[0]) if pub_date else 0
                    if year >= start_year_int:
                        primary_results.append(self.format_result(result))
                    else:
                        fallback_results.append(self.format_result(result))
                except ValueError:
                    fallback_results.append(self.format_result(result))
            
            logger.info(f"FDA results: {len(primary_results)}")
            return primary_results, fallback_results
        
        except Exception as e:
            logger.error(f"Error querying FDA API: {str(e)}")
            return [], []

    def rank_results(self, query, results, prompt_params):
        if not results:
            return []
        
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            context = "\n".join([f"Title: {r['title']}\nSummary: {r.get('summary', '')}\nDate: {r.get('publication_date', 'N/A')}" for r in results])
            prompt = f"""
            You are a research assistant tasked with ranking FDA articles based on their relevance to the query: "{query}". 
            For each article, provide:
            - The index of the article in the provided list (0 to {len(results)-1}).
            - A brief explanation of why the article is relevant to the query, considering the title, summary, and publication date.
            Return the ranking as a JSON array of objects with "index" and "explanation" fields, ordered from most to least relevant.
            Limit the ranking to the top {prompt_params.get('display_result_count', 80)} articles.
            Ensure the response is valid JSON and contains only the array of ranking objects.
            """
            
            completion = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": context}
                ],
                max_tokens=2000
            )
            
            response_text = completion.choices[0].message.content.strip()
            logger.info(f"Grok ranking response for fda: {response_text[:200]}...")
            
            # Remove code block markers if present
            if response_text.startswith('```json'):
                response_text = response_text[7:].rstrip('```')
            elif response_text.startswith('```'):
                response_text = response_text[3:].rstrip('```')
            
            ranked_indices = json.loads(response_text)
            
            valid_indices = {i for i in range(len(results))}
            ranked_results = []
            seen_indices = set()
            
            for item in ranked_indices:
                index = item.get('index')
                if index in valid_indices and index not in seen_indices:
                    ranked_results.append(results[index])
                    seen_indices.add(index)
            
            for i, result in enumerate(results):
                if i not in seen_indices:
                    ranked_results.append(result)
            
            logger.info(f"fda ranked {len(ranked_results)} results: indices {[r.get('index', i) for i, r in enumerate(ranked_results)]}")
            return ranked_results
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Grok ranking response for fda: {str(e)}, response: {response_text}")
            return results
        except Exception as e:
            logger.error(f"Error ranking FDA results: {str(e)}")
            return results

    def generate_summary(self, query, results, prompt_text, prompt_params):
        if not results or not prompt_text:
            return "No summary generated due to missing results or prompt."
        
        try:
            api_key = os.environ.get('XAI_API_KEY')
            if not api_key:
                raise ValueError("XAI_API_KEY not set")
            
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            context = "\n".join([f"Title: {r['title']}\nSummary: {r.get('summary', '')}\nDate: {r.get('publication_date', 'N/A')}" for r in results[:prompt_params.get('summary_result_count', 20)]])
            
            completion = client.chat.completions.create(
                model="grok-3",
                messages=[
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": context}
                ],
                max_tokens=1000
            )
            
            summary = completion.choices[0].message.content.strip()
            logger.info(f"Generated fda summary: length={len(summary)}")
            return summary
        
        except Exception as e:
            logger.error(f"Error generating FDA summary: {str(e)}")
            return f"Unable to generate summary: {str(e)}"

    def format_result(self, result):
        return {
            "id": result.get("id", ""),
            "title": result.get("title", "No title"),
            "summary": result.get("summary", ""),
            "authors": "N/A",
            "journal": "FDA.gov",
            "publication_date": result.get("publication_date", "N/A"),
            "url": result.get("url", "")
        }

class GoogleScholarSearchHandler(SearchHandler):
    def __init__(self):
        super().__init__(source_id="googlescholar", name="Google Scholar")
        self.proxy = os.environ.get('SCRAPER_API_KEY', None)

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
        retries = Retry(total=7, backoff_factor=3, status_forcelist=[403, 429], allowed_methods=["GET"])
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
            proxies = None
            if self.proxy:
                proxies = {
                    'https': f'http://scraperapi:{self.proxy}@proxy-server.scraperapi.com:8001'
                }
            
            time.sleep(random.uniform(2, 5))  # Randomized delay
            response = session.get(base_url, params=params, headers=headers, proxies=proxies, timeout=20, verify=False if self.proxy else True)
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

def esearch(query, max_results=80):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "api_key": os.environ.get('PUBMED_API_KEY', '')
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        return pmids
    except Exception as e:
        logger.error(f"Error in PubMed ESearch: {str(e)}")
        return []

def efetch(pmids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "api_key": os.environ.get('PUBMED_API_KEY', '')
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=20)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error in PubMed EFetch: {str(e)}")
        return ""

def parse_e_fetch_xml(xml_data):
    try:
        root = ET.fromstring(xml_data)
        articles = []
        
        for article in root.findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else ""
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else "No title"
            
            abstract_elem = article.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else ""
            
            authors = []
            for author in article.findall(".//Author"):
                last_name = author.find("LastName")
                initials = author.find("Initials")
                if last_name is not None and initials is not None:
                    authors.append(f"{last_name.text}, {initials.text}")
            authors_str = "; ".join(authors) if authors else "N/A"
            
            journal_elem = article.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else "N/A"
            
            pub_date = article.find(".//PubDate")
            year = pub_date.find("Year").text if pub_date is not None and pub_date.find("Year") is not None else ""
            month = pub_date.find("Month").text if pub_date is not None and pub_date.find("Month") is not None else ""
            day = pub_date.find("Day").text if pub_date is not None and pub_date.find("Day") is not None else ""
            pub_date_str = f"{year}-{month}-{day}" if year and month and day else year
            
            articles.append({
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors_str,
                "journal": journal,
                "publication_date": pub_date_str
            })
        
        return articles
    except Exception as e:
        logger.error(f"Error parsing PubMed XML: {str(e)}")
        return []

def search_fda_label_api(query, limit=80):
    base_url = "https://api.fda.gov/drug/label.json"
    params = {
        "search": query,
        "limit": limit
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        results = data.get('results', [])
        
        formatted_results = []
        for item in results:
            formatted_results.append({
                "id": item.get('id', ''),
                "title": item.get('openfda', {}).get('brand_name', ['No title'])[0],
                "summary": item.get('indications_and_usage', [''])[0],
                "publication_date": item.get('effective_time', '')[:8],
                "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={item.get('id', '')}"
            })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error querying FDA API: {str(e)}")
        return []

def extract_keywords_and_date(query, search_older=False, start_year=None):
    tokens = word_tokenize(query.lower())
    stop_words = set(stopwords.words('english'))
    tagged = pos_tag(tokens)
    
    keywords = []
    for word, pos in tagged:
        if pos.startswith('NN') or pos.startswith('JJ'):
            if word not in stop_words and len(word) > 2:
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name().lower() != word and '_' not in lemma.name():
                            synonyms.append(lemma.name().lower())
                keywords.append((word, list(set(synonyms))[:3]))
    
    date_range = None
    start_year_int = 2000
    today = datetime.now()
    
    if not search_older:
        start_date = today.replace(month=1, day=1).strftime('%Y/%m/%d')
        end_date = today.replace(month=12, day=31).strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
    elif start_year:
        try:
            start_year_int = int(start_year)
            start_date = f"{start_year_int}/01/01"
            end_date = today.strftime('%Y/%m/%d')
            date_range = f"{start_date}[dp]:{end_date}[dp]"
        except ValueError:
            logger.warning(f"Invalid start year: {start_year}, using default 2000")
            start_year_int = 2000
            start_date = "2000/01/01"
            end_date = today.strftime('%Y/%m/%d')
            date_range = f"{start_date}[dp]:{end_date}[dp]"
    
    logger.info(f"Extracted keywords: {keywords}, Date range: {date_range}")
    return keywords, date_range, start_year_int

def build_pubmed_query(keywords_with_synonyms, date_range):
    terms = []
    for keyword, synonyms in keywords_with_synonyms:
        all_terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        terms.append(f"({' OR '.join(all_terms)})")
    
    query = f"({' AND '.join(terms)})"
    if date_range:
        query += f" AND {date_range}"
    
    logger.info(f"Built PubMed query: {query}")
    return query