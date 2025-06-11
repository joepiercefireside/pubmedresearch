from flask import render_template, request, Response, session, make_response, current_app
from flask_login import login_required, current_user
import psycopg2
import json
import hashlib
import re
import time
import sqlite3
from datetime import datetime
from scipy.spatial.distance import cosine
from utils import esearch, efetch, parse_efetch_xml, extract_keywords_and_date, build_pubmed_query, PubMedSearchHandler, GoogleScholarSearchHandler
from core import app, logger, update_search_progress, query_grok_api, generate_embedding, get_cached_embedding, cache_embedding, get_db_connection, get_cached_grok_response, cache_grok_response

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

def parse_prompt(prompt_text):
    if not prompt_text:
        return {
            'summary_result_count': 3,
            'display_result_count': 20,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    summary_result_count = 3
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        summary_result_count = min(int(match.group(1)), 10)
    elif 'top' in prompt_text_lower:
        summary_result_count = 3
    
    display_result_count = 20
    limit_presentation = ('show only' in prompt_text_lower or 'present only' in prompt_text_lower)
    
    logger.info(f"Parsed prompt: summary_result_count={summary_result_count}, display_result_count={display_result_count}, limit_presentation={limit_presentation}")
    
    return {
        'summary_result_count': summary_result_count,
        'display_result_count': display_result_count,
        'limit_presentation': limit_presentation
    }

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

def generate_prompt_output(query, results, prompt_text, prompt_params, is_fallback=False):
    if not results:
        return f"No results found for '{query}'{' outside the specified timeframe' if is_fallback else ''}."
    
    logger.info(f"Initial results count: {len(results)}, is_fallback: {is_fallback}")
    
    summary_result_count = prompt_params.get('summary_result_count', 3) if prompt_params else 3
    context_results = results[:min(summary_result_count, 5)]
    logger.info(f"Context results count for summary: {len(context_results)}")
    
    if not context_results:
        return f"No results found for '{query}'{' outside the specified timeframe' if is_fallback else ''} matching criteria."
    
    context = "\n".join([f"Source: {r.get('source_id', 'unknown')}\nTitle: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nJournal: {r.get('journal', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nURL: {r.get('url', 'N/A')}" for r in context_results])
    
    MAX_CONTEXT_LENGTH = 8000
    if len(context) > MAX_CONTEXT_LENGTH:
        context = context[:MAX_CONTEXT_LENGTH] + "... [truncated]"
        logger.warning(f"Context truncated to {MAX_CONTEXT_LENGTH} characters for query: {query[:50]}...")
    
    try:
        cache_key = hashlib.md5((query.lower().strip() + context + prompt_text).encode()).hexdigest()
        cached_response = get_cached_grok_response(cache_key)
        if cached_response:
            output = cached_response
        else:
            strict_prompt = f"""
{prompt_text}

**Instructions**:
- Provide the response in Markdown format with exactly three paragraphs, one for each article summary.
- Separate paragraphs with a single blank line.
- Include hyperlinks to article URLs using [Article Title](URL).
- Use bold (**text**) for key terms and bullet points for key findings where applicable.
- Complete the response fully, ensuring all requested information is included.
- Do not include additional text beyond the three summaries unless explicitly requested.
"""
            output = query_grok_api(strict_prompt, context)
            logger.info(f"Raw Grok response for summary: {output[:200]}...")
            cache_grok_response(cache_key, output)
        
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\d+\.\s+', output) if p.strip()]
        if len(paragraphs) != 3:
            logger.warning(f"Grok returned {len(paragraphs)} paragraphs instead of 3")
            sentences = sent_tokenize(output)
            if len(sentences) >= 3:
                avg_len = len(sentences) // 3
                paragraphs = [
                    ' '.join(sentences[:avg_len]),
                    ' '.join(sentences[avg_len:2*avg_len]),
                    ' '.join(sentences[2*avg_len:])
                ]
            else:
                paragraphs = paragraphs[:3]
                paragraphs.extend(['No summary available.'] * (3 - len(paragraphs)))
        
        def markdown_to_html(text):
            text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
            text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', text)
            text = re.sub(r'^- (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
            if '<li>' in text:
                text = f'<ul>{text}</ul>'
            return text
        
        formatted_output = '\n'.join(f'<p>{markdown_to_html(p)}</p>' for p in paragraphs)
        logger.info(f"Generated prompt output: length={len(formatted_output)}, is_fallback: {is_fallback}")
        return formatted_output
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        output = f"Fallback: Unable to generate AI summary due to error: {str(e)}. Top results include: " + "; ".join([f"[{r['title']}]({r['url']}) ({r['publication_date']})" for r in context_results])
        paragraphs = output.split('; ')
        formatted_output = '\n'.join(f'<p>{markdown_to_html(p)}</p>' for p in paragraphs[:3])
        return formatted_output

@app.route('/search_progress', methods=['GET'])
@login_required
def search_progress():
    def stream_progress(user_id, query):
        with app.app_context():
            try:
                last_status = None
                while True:
                    conn = sqlite3.connect('search_progress.db')
                    c = conn.cursor()
                    try:
                        c.execute("SELECT status, timestamp FROM search_progress WHERE user_id = ? AND query = ? ORDER BY timestamp DESC LIMIT 1",
                                  (user_id, query))
                        result = c.fetchone()
                        if result and result[0] != last_status:
                            escaped_status = result[0].replace("'", "\\'")
                            yield 'data: {"status": "' + escaped_status + '"}\n\n'
                            last_status = result[0]
                        if result and result[0].startswith(("complete", "error")):
                            break
                    except sqlite3.Error as e:
                        logger.error(f"Error in search_progress: {str(e)}")
                        escaped_error = str(e).replace("'", "\\'")
                        yield 'data: {"status": "error: ' + escaped_error + '"}\n\n'
                        break
                    finally:
                        c.close()
                        conn.close()
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error in search_progress stream: {str(e)}")
                yield 'data: {"status": "error: ' + str(e).replace("'", "\\'") + '"}\n\n'
    
    query = request.args.get('query', '')
    response = Response(stream_progress(current_user.id, query), mimetype='text/event-stream')
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE user_id = %s', (current_user.id,))
        prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2]} for p in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        prompts = []
    finally:
        cur.close()
        conn.close()
    logger.info(f"Loaded prompts: {len(prompts)} prompts for user {current_user.id}")
    session.pop('latest_search_result_ids', None)

    page = {source: int(request.args.get(f'page_{source}', 1)) for source in ['pubmed', 'googlescholar']}
    per_page = 20
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'relevance'))
    prompt_id = request.form.get('prompt_id', request.args.get('prompt_id', ''))
    prompt_text = request.form.get('prompt_text', request.args.get('prompt_text', ''))
    query = request.form.get('query', request.args.get('query', ''))
    search_older = request.form.get('search_older', 'off') == 'on' or request.args.get('search_older', 'False') == 'True'
    start_year = request.form.get('start_year', request.args.get('start_year', None))
    if start_year == "None" or not start_year:
        start_year = None
    else:
        try:
            start_year = int(start_year)
        except ValueError:
            start_year = None
    sources_selected = request.form.getlist('sources') or request.args.getlist('sources') or []
    logger.debug(f"Sources selected: type={type(sources_selected)}, value={sources_selected}")

    selected_prompt_text = prompt_text
    if prompt_id and not prompt_text:
        for prompt in prompts:
            if str(prompt['id']) == prompt_id:
                selected_prompt_text = prompt['prompt_text']
                break
        else:
            logger.warning(f"Prompt ID {prompt_id} not found in prompts")
            selected_prompt_text = ''

    session['latest_prompt_text'] = selected_prompt_text

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50] if prompt_text else 'None'}..., query={query[:50] if query else 'None'}..., search_older={search_older}, start_year={start_year}, sources={sources_selected}, page={page}, sort_by={sort_by}")

    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'googlescholar': GoogleScholarSearchHandler()
    }

    if request.method == 'POST' or (request.method == 'GET' and query and sources_selected):
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            response = make_response(render_template('search.html', error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        if not sources_selected:
            update_search_progress(current_user.id, query, "error: At least one search source must be selected")
            response = make_response(render_template('search.html', error="At least one search source must be selected", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        update_search_progress(current_user.id, query, "Searching articles")

        try:
            keywords_with_synonyms, date_range, start_year_int = extract_keywords_and_date(query, search_older, start_year)
            if not keywords_with_synonyms:
                update_search_progress(current_user.id, query, "error: No valid keywords found")
                response = make_response(render_template('search.html', error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, 
                                       prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                       username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                       summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                       pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response

            prompt_params = parse_prompt(selected_prompt_text) or {}
            prompt_params['sort_by'] = sort_by
            summary_result_count = prompt_params.get('summary_result_count', 3)

            sources = []
            total_results = {}
            pubmed_results = []
            googlescholar_results = []
            pubmed_fallback_results = []
            all_results = []
            all_ranked_results = []

            for source_id in sources_selected:
                if source_id not in search_handlers:
                    logger.warning(f"Unknown source: {source_id}")
                    continue

                handler = search_handlers[source_id]
                update_search_progress(current_user.id, query, f"Searching {handler.name}")

                primary_results, fallback_results = handler.search(query, keywords_with_synonyms, date_range, start_year_int)
                update_search_progress(current_user.id, query, f"Found {len(primary_results)} {handler.name} PMIDs")

                primary_results = primary_results or [][:20]
                fallback_results = fallback_results or [][:20]

                ranked_results = []
                if primary_results:
                    update_search_progress(current_user.id, query, f"Ranking {handler.name} results")
                    ranked_results = rank_results(query, primary_results, prompt_params)

                source_summary = ""
                if selected_prompt_text and ranked_results:
                    update_search_progress(current_user.id, query, f"Generating {handler.name} summary")
                    source_summary = generate_prompt_output(query, ranked_results, selected_prompt_text, prompt_params)

                source_data = {
                    'id': handler.source_id,
                    'name': handler.name,
                    'results': {
                        'ranked': ranked_results,
                        'all': primary_results,
                        'fallback': fallback_results
                    },
                    'summary': source_summary
                }

                if source_id == 'pubmed':
                    if primary_results or fallback_results:
                        conn = get_db_connection()
                        cur = conn.cursor()
                        try:
                            cur.execute("INSERT INTO search_cache (query, results, created_at) VALUES (%s, %s, %s)", 
                                        (query, json.dumps(primary_results + fallback_results), datetime.now()))
                            conn.commit()
                        except Exception as e:
                            logger.error(f"Error caching search results: {str(e)}")
                            conn.rollback()
                        finally:
                            cur.close()
                            conn.close()
                        pubmed_results = primary_results
                        pubmed_fallback_results = fallback_results
                elif source_id == 'googlescholar':
                    googlescholar_results = primary_results

                source_total = len(primary_results) + len(fallback_results)
                total_results[source_id] = source_total
                sources.append(source_data)

                all_results.extend([dict(r, source_id=source_id) for r in primary_results])
                all_results.extend([dict(r, source_id=source_id) for r in fallback_results])
                all_ranked_results.extend([dict(r, source_id=source_id) for r in ranked_results[:summary_result_count]])

            total_pages = {source_id: (total_results.get(source_id, 0) + per_page - 1) // per_page for source_id in total_results}

            for source in sources:
                source_page = page.get(source['id'], 1)
                start_idx = (source_page - 1) * per_page
                end_idx = start_idx + per_page
                source['results']['ranked'] = source['results']['ranked'][start_idx:end_idx]
                source['results']['all'] = source['results']['all'][start_idx:end_idx]
                source['results']['fallback'] = source['results']['fallback'][start_idx:end_idx]

            result_ids = save_search_results(current_user.id, query, all_results)
            session['latest_search_result_ids'] = json.dumps(result_ids[:10])
            session['latest_query'] = query

            update_search_progress(current_user.id, query, "complete")

            logger.debug("Rendering search template for POST/GET request")
            response = make_response(render_template(
                'search.html', 
                sources=sources,
                total_results=total_results,
                total_pages=total_pages,
                page=page,
                per_page=per_page,
                query=query, 
                prompts=prompts, 
                prompt_id=prompt_id,
                prompt_text=selected_prompt_text,
                summary_result_count=summary_result_count,
                username=current_user.email,
                has_prompt=bool(selected_prompt_text),
                prompt_params=prompt_params,
                search_older=search_older,
                start_year=start_year,
                sort_by=sort_by,
                pubmed_results=pubmed_results,
                googlescholar_results=googlescholar_results,
                pubmed_fallback_results=pubmed_fallback_results,
                sources_selected=sources_selected,
                combined_summary=''
            ))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        except Exception as e:
            logger.error(f"API error in POST/GET: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            response = make_response(render_template('search.html', error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id, 
                                   prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                                   username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                                   summary_result_count=3, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                                   pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary=''))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

    logger.debug("Rendering search template for GET request")
    response = make_response(render_template(
        'search.html', 
        prompts=prompts, 
        prompt_id=prompt_id, 
        prompt_text=selected_prompt_text, 
        sources=[],
        total_results={},
        total_pages={},
        page=page,
        per_page=per_page,
        username=current_user.email, 
        has_prompt=bool(selected_prompt_text), 
        prompt_params={}, 
        summary_result_count=3, 
        search_older=False, 
        start_year=None,
        sort_by=sort_by,
        pubmed_results=[],
        pubmed_fallback_results=[],
        sources_selected=sources_selected,
        combined_summary=''
    ))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response