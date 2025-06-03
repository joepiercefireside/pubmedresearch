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

logger = logging.getLogger(__name__)

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
        try:
            articles_context = []
            for i, result in enumerate(results):
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
            ranking = json.loads(response)
            
            ranked_indices = []
            for item in ranking:
                if isinstance(item, dict) and 'index' in item:
                    index = int(item['index']) - 1
                    if 0 <= index < len(results):
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
            search_terms.extend([t.strip() for t in terms if t.strip()])
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
    stop_words = {
        'and', 'or', 'not', 'from', 'the', 'past', 'years', 'about', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'related', 'what', 'can', 'you', 'tell', 'me', 'is', 'are', 
        'how', 'why', 'when', 'where', 'who', 'which'
    }
    timeframe_patterns = [
        (r'within\s+the\s+last\s+(\d+)\s+year(s?)', lambda m: int(m.group(1))),
        (r'in\s+(\d{4})', lambda m: (int(m.group(1)), int(m.group(1)))),
        (r'since\s+(\d{4})', lambda m: (int(m.group(1)), datetime.now().year)),
        (r'from\s+(\d{4})\s+to\s+(\d{4})', lambda m: (int(m.group(1)), int(m.group(2))))
    ]
    
    query_lower = query.lower()
    start_year_int = 2000
    date_range = None
    
    # Parse timeframe from query
    for pattern, extractor in timeframe_patterns:
        if match := re.search(pattern, query_lower):
            result = extractor(match)
            if isinstance(result, tuple):
                start_year, end_year = result
            else:
                start_year = datetime.now().year - result
                end_year = datetime.now().year
            start_date = f"{start_year}/01/01"
            end_date = f"{end_year}/12/31"
            date_range = f"{start_date}[dp]:{end_date}[dp]"
            start_year_int = start_year
            break
    
    # Default to 5 years if no timeframe specified
    if not date_range and not search_older:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y/%m/%d')
        end_date = datetime.now().strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_date[:4])
    elif search_older and start_year:
        start_date = f"{start_year}/01/01"
        end_date = datetime.now().strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_year)
    
    # Extract keywords and synonyms
    words = query_lower.split()
    keywords = [(word, []) for word in words if word not in stop_words and not word.isdigit()]
    
    # Simple synonym mapping (expandable)
    synonym_map = {
        'diabetes': ['diabetic', 'glucose', 'insulin'],
        'treatment': ['therapy', 'management', 'care'],
        'weight': ['obesity', 'body mass', 'fat'],
        'new': ['recent', 'novel', 'latest']
    }
    for i, (keyword, _) in enumerate(keywords):
        keywords[i] = (keyword, synonym_map.get(keyword, []))
    
    logger.info(f"Extracted keywords: {keywords}, Date range: {date_range}")
    return keywords, date_range, start_year_int

def build_pubmed_query(keywords_with_synonyms, date_range):
    if not keywords_with_synonyms:
        return ""
    terms = []
    for keyword, synonyms in keywords_with_synonyms:
        all_terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        terms.append("(" + " OR ".join(all_terms) + ")")
    query = " AND ".join(terms)
    if date_range:
        query += f" AND {date_range}"
    logger.info(f"Built PubMed query: {query}")
    return query