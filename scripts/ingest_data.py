import ftplib
import os
from urllib.request import urlopen
import gzip
import psycopg2
from psycopg2.extras import execute_values
import xml.etree.ElementTree as ET

def download_baseline():
    ftp_server = "ftp.ncbi.nlm.nih.gov"
    ftp_dir = "/pubmed/baseline/"
    ftp = ftplib.FTP(ftp_server)
    try:
        ftp.login()
        ftp.cwd(ftp_dir)
        files = [f for f in ftp.nlst() if f.endswith('.gz')]
        ftp.quit()
        return files
    except ftplib.error_perm as e:
        print(f"FTP error: {e}")
        print("Available directories:")
        ftp.cwd("/pubmed/")
        print(ftp.nlst())
        ftp.quit()
        raise

def parse_article(file_path):
    articles = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for article in root.findall('.//MedlineCitation'):
            pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else ''
            title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else ''
            abstract = ''.join(article.find('.//AbstractText').itertext()) if article.find('.//AbstractText') is not None else ''
            # Handle publication date
            pub_date = None
            year = article.find('.//PubDate/Year')
            if year is not None and year.text:
                try:
                    # Try to convert year to YYYY-01-01
                    pub_date = f"{int(year.text):04d}-01-01"
                except (ValueError, TypeError):
                    pub_date = None  # Invalid year, set to NULL
            authors = ', '.join([au.find('LastName').text for au in article.findall('.//Author') if au.find('LastName') is not None])
            journal = article.find('.//Journal/Title').text if article.find('.//Journal/Title') is not None else ''
            articles.append((pmid, title, abstract, pub_date, authors, journal))
    return articles

def ingest_to_db(articles):
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    query = """
    INSERT INTO articles (pmid, title, abstract, publication_date, authors, journal)
    VALUES %s
    ON CONFLICT (pmid) DO NOTHING
    """
    execute_values(cur, query, articles)
    conn.commit()
    cur.close()
    conn.close()

def main():
    files = download_baseline()
    for file in files[:5]:  # Process one file for testing
        url = f"ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/{file}"
        local_file = file
        with urlopen(url) as response, open(local_file, 'wb') as out_file:
            out_file.write(response.read())
        articles = parse_article(local_file)
        ingest_to_db(articles)
        os.remove(local_file)

if __name__ == "__main__":
    main()