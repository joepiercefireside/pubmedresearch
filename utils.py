import requests
from urllib.parse import quote
import logging
import re
from xml.etree import ElementTree
from ratelimit import limits, sleep_and_retry
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from datetime import datetime, timedelta

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
    "hypertension": ["high blood pressure", "elevated BP"]
}

@sleep_and_retry
@limits(calls=10, period=1)
def esearch(query, retmax=80, api_key=None):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "sort": "relevance",
        "api_key": api_key
    }
    logger.info(f"PubMed ESearch query: {query}")
    response = requests.get(url, params=params)
    response.raise_for_status()
    result = response.json()
    logger.info(f"ESearch result: {len(result['esearchresult']['idlist'])} PMIDs")
    return result

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
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.content

def parse_efetch_xml(xml_content):
    root = ElementTree.fromstring(xml_content)
    articles = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.find(".//PMID").text if article.find(".//PMID") is not None else ""
        title = article.find(".//ArticleTitle").text if article.find(".//ArticleTitle") is not None else ""
        abstract = article.find(".//AbstractText")
        abstract = abstract.text or "" if abstract is not None else ""
        authors = [author.find("LastName").text for author in article.findall(".//Author") 
                   if author.find("LastName") is not None]
        journal = article.find(".//Journal/Title")
        journal = journal.text if journal is not None else ""
        pub_date = article.find(".//PubDate/Year")
        pub_date = pub_date.text if pub_date is not None else ""
        logger.info(f"Parsed article: PMID={pmid}, Date={pub_date}, Abstract={'Present' if abstract else 'Missing'}")
        articles.append({
            "id": pmid,
            "title": title,
            "abstract": abstract,
            "authors": ", ".join(authors),
            "journal": journal,
            "publication_date": pub_date
        })
    return articles

def search_fda_api(query, keywords_with_synonyms, date_range, api_key=None):
    try:
        search_terms = []
        for keyword, synonyms in keywords_with_synonyms:
            terms = [keyword] + synonyms
            search_terms.extend(terms)
        search_query = quote(' '.join(search_terms))
        
        if date_range:
            start_date = date_range.split(' TO ')[0].replace('/', '')[:8]
            end_date = date_range.split(' TO ')[1].replace('/', '')[:8]
            date_filter = f"&search=receivedate:[+{start_date}+TO+{end_date}]"
        else:
            date_filter = ""
        
        url = f"https://api.fda.gov/drug/event.json?search={search_query}{date_filter}&limit=20"
        headers = {'User-Agent': 'PubMedResearch/1.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get('results', []):
            # Safely access reaction field
            reaction = item.get('reaction', [])
            title = reaction[0].get('reactionmeddrapt', 'No title') if reaction else 'No title'
            results.append({
                'id': item.get('safetyreportid', ''),
                'title': title,
                'summary': item.get('patient', {}).get('patientnarrative', ''),
                'date': item.get('receivedate', '')[:8] or 'N/A',
                'url': f"https://open.fda.gov/data/faers/{item.get('safetyreportid', '')}"
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
    stop_words = set(stopwords.words('english')).union({'what', 'can', 'tell', 'me', 'is', 'new', 'in', 'the', 'of', 'for', 'any', 'articles', 'that', 'show', 'between', 'only', 'related', 'to', 'available'})
    
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
        date_range = f"{start_year}/01/01[dp]:{today.strftime('%Y/%m/%d')}[dp]"
    elif year_match := re.search(r'\b(20\d{2})\b', query_lower):
        year = int(year_match.group(1))
        date_range = f"{year}/01/01[dp]:{year}/12/31[dp]"
    elif 'past year' in query_lower:
        date_range = f"{(today - timedelta(days=365)).strftime('%Y/%m/%d')}[dp]:{today.strftime('%Y/%m/%d')}[dp]"
    elif 'past week' in query_lower:
        date_range = f"{(today - timedelta(days=7)).strftime('%Y/%m/%d')}[dp]:{today.strftime('%Y/%m/%d')}[dp]"
    else:
        start_year_int = int(start_year) if search_older and start_year else default_start_year
        date_range = f"{start_year_int}/01/01[dp]:{today.strftime('%Y/%m/%d')}[dp]"
    
    logger.info(f"Extracted keywords: {keywords_with_synonyms}, Date range: {date_range}")
    return keywords_with_synonyms, date_range, start_year_int if search_older and start_year else default_start_year

def build_pubmed_query(keywords_with_synonyms, date_range):
    query_parts = []
    for keyword, synonyms in keywords_with_synonyms:
        terms = [f'"{keyword}"[All Fields]'] + [f'"{syn}"[All Fields]' for syn in synonyms]
        term_query = " OR ".join(terms)
        query_parts.append(f"({term_query})")
    
    query = " AND ".join(query_parts) if query_parts else ""
    query = f"({query}) AND {date_range}" if query else date_range
    
    logger.info(f"Built PubMed query: {query}")
    return query