import ftplib
import os
from urllib.request import urlopen
import gzip
import psycopg2
from psycopg2.extras import execute_values
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time

def download_baseline():
    """List all .gz files in PubMed baseline directory."""
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

def download_file(ftp_server, ftp_dir, filename, local_dir="data"):
    """Download a file from FTP with resume support."""
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, filename)
    ftp = ftplib.FTP(ftp_server)
    ftp.login()
    ftp.cwd(ftp_dir)
    
    # Check local file size for resuming
    local_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
    ftp.voidcmd("TYPE I")  # Binary mode
    remote_size = ftp.size(filename)
    
    if local_size == remote_size:
        print(f"File {filename} already fully downloaded.")
        ftp.quit()
        return local_path
    
    with open(local_path, 'ab' if local_size > 0 else 'wb') as f:
        if local_size > 0:
            print(f"Resuming download of {filename} from {local_size} bytes...")
            ftp.retrbinary(f"RETR {filename}", f.write, rest=local_size)
        else:
            print(f"Downloading {filename}...")
            ftp.retrbinary(f"RETR {filename}", f.write)
    
    ftp.quit()
    return local_path

def parse_article(file_path):
    """Parse PubMed XML file and extract article data."""
    articles = []
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            for article in root.findall('.//MedlineCitation'):
                pmid = article.find('.//PMID').text if article.find('.//PMID') is not None else ''
                title = article.find('.//ArticleTitle').text if article.find('.//ArticleTitle') is not None else ''
                abstract_elem = article.find('.//AbstractText')
                abstract = ''.join(abstract_elem.itertext()) if abstract_elem is not None else ''
                # Handle publication date
                pub_date = None
                year = article.find('.//PubDate/Year')
                if year is not None and year.text:
                    try:
                        pub_date = f"{int(year.text):04d}-01-01"
                    except (ValueError, TypeError):
                        pub_date = None
                authors = ', '.join([au.find('LastName').text for au in article.findall('.//Author') if au.find('LastName') is not None])
                journal = article.find('.//Journal/Title').text if article.find('.//Journal/Title') is not None else ''
                articles.append((pmid, title, abstract, pub_date, authors, journal))
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return []
    return articles

def ingest_to_db(articles):
    """Insert articles into PostgreSQL database."""
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
    """Download and ingest all PubMed baseline files with progress reporting."""
    ftp_server = "ftp.ncbi.nlm.nih.gov"
    ftp_dir = "/pubmed/baseline/"
    files = download_baseline()
    print(f"Found {len(files)} files to process.")
    
    total_articles = 0
    start_time = time.time()
    
    for file in tqdm(files, desc="Processing files"):
        try:
            local_file = download_file(ftp_server, ftp_dir, file)
            articles = parse_article(local_file)
            if articles:
                ingest_to_db(articles)
                total_articles += len(articles)
                print(f"Processed {file}: {len(articles)} articles (Total: {total_articles})")
            else:
                print(f"No articles extracted from {file}")
            os.remove(local_file)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Completed ingestion: {total_articles} articles in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    os.environ['DATABASE_URL'] = "your_database_url_here"  # Replace with actual DATABASE_URL
    main()