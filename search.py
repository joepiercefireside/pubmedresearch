from flask import render_template, request, Response, session, make_response, current_app
from flask_login import login_required, current_user
import json
import hashlib
import time
import psycopg2
from datetime import datetime
from utils import esearch, efetch, parse_efetch_xml, extract_keywords_and_date, build_pubmed_query, PubMedSearchHandler, GoogleScholarSearchHandler, SemanticScholarSearchHandler
from core import app, logger, update_search_progress, query_grok_api, get_db_connection, get_cached_grok_response, cache_grok_response
from search_utils import save_search_results, get_search_results, rank_results
from features import save_search_history
from prompt_utils import parse_prompt
import nltk
from nltk.tokenize import sent_tokenize
import mistune

# ... (other existing code remains unchanged until the search function)

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
    
    # Clear old session result_limit to avoid persistence issues
    if 'result_limit' in session:
        del session['result_limit']

    page = {source: int(request.args.get(f'page_{source}', 1)) for source in ['pubmed', 'googlescholar', 'semanticscholar']}
    per_page = 20
    sort_by = request.form.get('sort_by', request.args.get('sort_by', 'relevance'))
    prompt_id = request.form.get('prompt_id', request.args.get('prompt_id', ''))
    prompt_text = request.form.get('prompt_text', request.args.get('prompt_text', ''))
    query = request.form.get('query', request.args.get('query', ''))

    # Calculate current year once at the start
    current_year = datetime.now().year

    # Log result_limit sources for debugging
    form_limit = request.form.get('result_limit')
    args_limit = request.args.get('result_limit')
    logger.debug(f"result_limit sources: form={form_limit}, args={args_limit}")
    result_limit = form_limit or args_limit or '20'
    try:
        result_limit = int(result_limit)
        if result_limit not in [10, 20, 50, 100]:
            result_limit = 20
    except ValueError:
        result_limit = 20
    session['result_limit'] = str(result_limit)  # Save to session
    logger.debug(f"Final result_limit: {result_limit}")

    search_older = request.form.get('search_older', 'off') == 'on' or request.args.get('search_older', 'False') == 'True'
    start_year = request.form.get('start_year', request.args.get('start_year', None))

    # Adjusted start_year logic
    if not search_older:
        start_year = current_year - 5  # Default to last 5 years
    else:
        if start_year and start_year != "None":
            try:
                start_year = int(start_year)
                if start_year < 1900 or start_year > current_year:
                    start_year = 1900
            except ValueError:
                start_year = 1900
        else:
            start_year = 1900

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

    logger.info(f"Search request: prompt_id={prompt_id}, prompt_text={prompt_text[:50] if prompt_text else 'None'}..., query={query[:50] if query else 'None'}..., search_older={search_older}, start_year={start_year}, sources={sources_selected}, page={page}, sort_by={sort_by}, result_limit={result_limit}")

    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'googlescholar': GoogleScholarSearchHandler(),
        'semanticscholar': SemanticScholarSearchHandler()
    }

    if request.method == 'POST' or (request.method == 'GET' and query and sources_selected):
        if not query:
            update_search_progress(current_user.id, query, "error: Query cannot be empty")
            response = make_response(render_template('search.html', 
                current_year=current_year,
                error="Query cannot be empty", prompts=prompts, prompt_id=prompt_id,
                prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page,
                username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={},
                summary_result_count=20, search_older=search_older, start_year=start_year, sort_by=sort_by,
                pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary='', result_limit=result_limit))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        if not sources_selected:
            update_search_progress(current_user.id, query, "error: At least one search source must be selected")
            response = make_response(render_template('search.html',
                current_year=current_year,
                error="At least one search source must be selected", prompts=prompts, prompt_id=prompt_id,
                prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page,
                username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={},
                summary_result_count=20, search_older=search_older, start_year=start_year, sort_by=sort_by,
                pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary='', result_limit=result_limit))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

        update_search_progress(current_user.id, query, "Searching articles")

        try:
            keywords_with_synonyms, date_range, start_year_int = extract_keywords_and_date(query, search_older, start_year)
            if date_range and 'start' in date_range:
                start_year_int = date_range['start']
            else:
                start_year_int = start_year

            if not keywords_with_synonyms:
                update_search_progress(current_user.id, query, "error: No valid keywords found")
                response = make_response(render_template('search.html',
                    current_year=current_year,
                    error="No valid keywords found", prompts=prompts, prompt_id=prompt_id,
                    prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page,
                    username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={},
                    summary_result_count=20, search_older=search_older, start_year=start_year, sort_by=sort_by,
                    pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary='', result_limit=result_limit))
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response

            prompt_params = parse_prompt(selected_prompt_text) or {}
            prompt_params['sort_by'] = sort_by
            summary_result_count = prompt_params.get('summary_result_count', 20)

            sources = []
            total_results = {}
            pubmed_results = []
            googlescholar_results = []
            semanticscholar_results = []
            pubmed_fallback_results = []
            all_results = []
            all_ranked_results = []

            for source_id in sources_selected:
                if source_id not in search_handlers:
                    logger.warning(f"Unknown source: {source_id}")
                    continue

                handler = search_handlers[source_id]
                update_search_progress(current_user.id, query, f"Searching {handler.name}")

                max_retries = 3
                retry_delay = 5  # seconds
                for attempt in range(max_retries):
                    try:
                        primary_results, fallback_results = handler.search(query, keywords_with_synonyms, date_range, start_year_int or current_year - 10, result_limit=result_limit)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1 and "429" in str(e):
                            logger.warning(f"Rate limit hit for {handler.name}, retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(retry_delay)
                            continue
                        raise

                if source_id == 'pubmed':
                    update_search_progress(current_user.id, query, f"Found {len(primary_results)} {handler.name} PMIDs")
                else:
                    update_search_progress(current_user.id, query, f"Found {len(primary_results)} {handler.name} results")

                primary_results = primary_results or [][:result_limit]
                fallback_results = fallback_results or [][:result_limit]

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
                elif source_id == 'semanticscholar':
                    semanticscholar_results = primary_results

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
            search_id = hashlib.sha256((query + str(time.time())).encode()).hexdigest()[:16]
            session['latest_search_id'] = search_id
            session['latest_search_result_ids'] = json.dumps(result_ids[:10])
            session['latest_query'] = query

            save_search_history(current_user.id, query, selected_prompt_text, sources_selected, all_results, search_id)

            update_search_progress(current_user.id, query, "complete")

            logger.debug("Rendering search template for POST/GET request")
            response = make_response(render_template(
                'search.html', 
                current_year=current_year,
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
                semanticscholar_results=semanticscholar_results,
                pubmed_fallback_results=pubmed_fallback_results,
                sources_selected=sources_selected,
                combined_summary='',
                result_limit=result_limit
            ))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        except Exception as e:
            logger.error(f"API error in POST/GET: {str(e)}")
            update_search_progress(current_user.id, query, f"error: Search failed: {str(e)}")
            response = make_response(render_template('search.html',
                current_year=current_year,
                error=f"Search failed: {str(e)}", prompts=prompts, prompt_id=prompt_id,
                prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page,
                username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={},
                summary_result_count=20, search_older=search_older, start_year=start_year, sort_by=sort_by,
                pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary='', result_limit=result_limit))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response

    logger.debug("Rendering search template for GET request")
    response = make_response(render_template(
        'search.html', 
        current_year=current_year,
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
        summary_result_count=20, 
        search_older=False, 
        start_year=None,
        sort_by=sort_by,
        pubmed_results=[],
        googlescholar_results=[],
        semanticscholar_results=[],
        pubmed_fallback_results=[],
        sources_selected=sources_selected,
        combined_summary='',
        result_limit=result_limit
    ))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response