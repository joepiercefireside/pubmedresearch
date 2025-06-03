import requests
import logging
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timedelta
from urllib.parse import quote

logger = logging.getLogger(__name__)

def esearch(query, retmax=100, api_key=None):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
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
        "id": ",".join(pmids),
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

def parse_efetch_xml(xml_data):
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

def search_fda_api(query, keywords_with_synonyms, date_range, api_key=None):
    try:
        search_terms = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [keyword] + synonyms
            search_terms.extend(terms)
        search_query = quote(' '.join(search_terms))
        
        date_filter = ""
        if date_range:
            start_date = date_range.split(':')[0].replace('/', '')
            end_date = date_range.split(':')[1].replace('[dp]', '').replace('/', '')
            date_filter = f"+AND+effective_time:[{start_date}+TO+{end_date}]"
        
        # Use drug label endpoint with corrected query syntax
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

def extract_keywords_and_date(query, search_older, start_year):
    stop_words = {'and', 'or', 'not', 'from', 'the', 'past', 'years', 'about', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'related'}
    words = query.lower().split()
    keywords = [(word, []) for word in words if word not in stop_words and not word.isdigit()]
    date_range = None
    start_year_int = 2000  # Default start year
    if not search_older:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y/%m/%d')
        end_date = datetime.now().strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_date[:4])
    elif start_year:
        start_date = f"{start_year}/01/01"
        end_date = datetime.now().strftime('%Y/%m/%d')
        date_range = f"{start_date}[dp]:{end_date}[dp]"
        start_year_int = int(start_year)
    logger.info(f"Extracted keywords: {keywords}, Date range: {date_range}")
    return keywords, date_range, start_year_int

def build_pubmed_query(keywords_with_synonyms, date_range):
    terms = []
    for keyword, synonyms in keywords_with_synonyms:
        all_terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        terms.append("(" + " OR ".join(all_terms) + ")")
    query = " AND ".join(terms)
    if date_range:
        query += f" AND {date_range}"
    logger.info(f"Built PubMed query: {query}")
    return query