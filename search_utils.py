import mistune
from core import app, logger, query_grok_api, get_cached_grok_response, cache_grok_response, generate_embedding, get_cached_embedding, cache_embedding, get_db_connection
import psycopg2
import hashlib
import json
from datetime import datetime

def markdown_to_html(text):
    return mistune.html(text)

def save_search_results(user_id, query, results):
    conn = get_db_connection()
    cur = conn.cursor()
    result_ids = []
    try:
        for result in results[:50]:
            result_id = hashlib.md5(json.dumps(result, sort_keys=True).encode()).hexdigest()
            result_data = json.dumps(result)
            cur.execute(
                "INSERT INTO search_results (user_id, query, source_id, result_data, created_at) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                (str(user_id), query, result.get('source_id', 'unknown'), result_data, datetime.now())
            )
            result_ids.append(result_id)
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving search results: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    return result_ids

def get_search_results(user_id, query):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT result_data FROM search_results WHERE user_id = %s AND query = %s ORDER BY created_at DESC LIMIT 10",
            (str(user_id), query)
        )
        results = []
        for row in cur.fetchall():
            try:
                result_data = row[0]
                if isinstance(result_data, str):
                    results.append(json.loads(result_data))
                else:
                    results.append(result_data)
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding JSON for search result: {str(e)}")
                continue
        return results
    except Exception as e:
        logger.error(f"Error retrieving search results: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def rank_results(query, results, prompt_params=None):
    display_result_count = prompt_params.get('display_result_count', 20) if prompt_params else 20
    try:
        articles_context = []
        for i, result in enumerate(results[:10]):
            article_text = f"Article {i+1}: Title: {result['title']}\nAbstract: {result.get('abstract', '')}\nAuthors: {result.get('authors', 'N/A')}\nJournal: {result.get('journal', 'N/A')}\nDate: {result.get('publication_date', 'N/A')}"
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
        cache_key = hashlib.md5((query.lower().strip() + context + ranking_prompt).encode()).hexdigest()
        cached_response = get_cached_grok_response(cache_key)
        if cached_response:
            response = cached_response
        else:
            response = query_grok_api(ranking_prompt, context)
            try:
                json.loads(response)
                cache_grok_response(cache_key, response)
            except json.JSONDecodeError as je:
                logger.error(f"Invalid JSON from Grok: {str(je)}")
                raise
        logger.info(f"Grok ranking response: {response[:200]}...")
        ranking = json.loads(response)
        if not isinstance(ranking, list):
            raise ValueError("Grok response is not a list")
        
        ranked_indices = []
        for item in ranking:
            if isinstance(item, dict) and 'index' in item:
                index = item['index']
                if isinstance(index, (int, str)) and str(index).isdigit():
                    index = int(index) - 1
                    if 0 <= index < len(results):
                        ranked_indices.append(index)
        
        missing_indices = [i for i in range(len(results)) if i not in ranked_indices]
        ranked_indices.extend(missing_indices)
        
        ranked_results = [results[i] for i in ranked_indices[:display_result_count]]
        logger.info(f"Grok ranked {len(ranked_results)} results: indices {ranked_indices[:display_result_count]}")
        return ranked_results
    except Exception as e:
        logger.error(f"Grok ranking failed: {str(e)}")
        return embedding_based_ranking(query, results, prompt_params)

def embedding_based_ranking(query, results, prompt_params=None):
    display_result_count = prompt_params.get('display_result_count', 20) if prompt_params else 20
    query_embedding = generate_embedding(query)
    if query_embedding is None:
        logger.error("Failed to generate query embedding")
        return results[:display_result_count]
    current_year = datetime.now().year
    
    results = results[:10]
    embeddings = []
    texts = []
    for result in results:
        pmid = result.get('url', '').split('/')[-2] if 'pubmed' in result.get('url', '') else f"{result['title']}_{result['publication_date']}"
        embedding = get_cached_embedding(pmid)
        if embedding is None:
            texts.append(f"{result['title']} {result.get('abstract', '')}")
        else:
            embeddings.append(embedding)
    
    if texts:
        model = load_embedding_model()
        new_embeddings = model.encode(texts, convert_to_numpy=True)
        for i, (result, emb) in enumerate(zip(results[len(embeddings):], new_embeddings)):
            pmid = result.get('url', '').split('/')[-2] if 'pubmed' in result.get('url', '') else f"{result['title']}_{result['publication_date']}"
            if emb.shape[0] == 384:
                cache_embedding(pmid, emb)
                embeddings.append(emb)
            else:
                logger.error(f"Generated embedding for {pmid} has incorrect dimension: {emb.shape[0]}")
                embeddings.append(None)
    
    scores = []
    for i, (emb, result) in enumerate(zip(embeddings, results)):
        if emb is not None and emb.shape[0] == 384:
            similarity = 1 - cosine(query_embedding, emb)
        else:
            similarity = 0.0
        pub_year = int(result['publication_date']) if result['publication_date'] and result['publication_date'].isdigit() else 2000
        recency_bonus = (pub_year - 2000) / (current_year - 2000)
        weighted_score = (0.8 * similarity) + (0.2 * recency_bonus)
        scores.append((i, weighted_score, pub_year))
    
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    ranked_indices = [i for i, _, _ in scores]
    
    ranked_results = [results[i] for i in ranked_indices[:display_result_count]]
    logger.info(f"Embedding-based ranked {len(ranked_results)} results with indices {ranked_indices[:display_result_count]}")
    return ranked_results