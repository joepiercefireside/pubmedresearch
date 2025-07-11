import ftplib
import gzip
import os
import xml.etree.ElementTree as ET
import psycopg2
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get('DATABASE_URL')

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

def download_updates():
    ftp_server = "ftp.ncbi.nlm.nih.gov"
    ftp_dir = "/pubmed/updatefiles"
    local_dir = "data/updates"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    ftp = ftplib.FTP(ftp_server)
    ftp.login()
    ftp.cwd(ftp_dir)
    files = ftp.nlst("*.xml.gz")
    latest_file = sorted(files)[-1]  # Get the latest update
    local_path = os.path.join(local_dir, latest_file)
    with open(local_path, 'wb') as f:
        ftp.retrbinary(f"RETR {latest_file}", f.write)
    ftp.quit()
    logger.info(f"Downloaded update {latest_file}")
    return local_path

def process_update(file_path):
    conn = get_db_connection()
    cur = conn.cursor()
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        for article in root.findall('.//MedlineCitation'):
            pmid = article.find('PMID').text
            title = article.find('.//ArticleTitle').text or ''
            abstract = ''
            abstract_elem = article.find('.//AbstractText')
            if abstract_elem is not None:
                abstract = abstract_elem.text or ''
            authors = []
            for author in article.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None and fore_name is not None:
                    authors.append({'last_name': last_name.text, 'fore_name': fore_name.text})
            keywords = []
            for keyword in article.findall('.//Keyword'):
                if keyword.text:
                    keywords.append(keyword.text)
            pub_date = article.find('.//PubDate')
            pub_year = pub_date.find('Year').text if pub_date.find('Year') is not None else '1900'
            pub_date = datetime.strptime(pub_year, '%Y').date()
            
            cur.execute("""
                INSERT INTO articles (pmid, title, abstract, authors, keywords, publication_date)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (pmid) DO UPDATE
                SET title = EXCLUDED.title,
                    abstract = EXCLUDED.abstract,
                    authors = EXCLUDED.authors,
                    keywords = EXCLUDED.keywords,
                    publication_date = EXCLUDED.publication_date
            """, (pmid, title, abstract, authors, keywords, pub_date))
        
        for delete in root.findall('.//DeleteCitation/PMID'):
            pmid = delete.text
            cur.execute("DELETE FROM articles WHERE pmid = %s", (pmid,))
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Processed update {file_path}")

if __name__ == '__main__':
    file_path = download_updates()
    process_update(file_path)