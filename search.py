from flask import render_template, request, Response, session, make_response, current_app
from flask_login import login_required, current_user
import json
import hashlib
import time
import psycopg2
from datetime import datetime  # Already imported, ensure it’s used
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
                if start_year < 1900 or start_year > current_year:  # Adjusted minimum year
                    start_year = 1900
            except ValueError:
                start_year = 1900  # Default to earliest year if invalid
        else:
            start_year = 1900  # "Any" option maps to earliest year

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
                current_year=current_year,  # Pass current_year
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
                current_year=current_year,  # Pass current_year
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
            # If date_range is extracted from query, it overrides start_year
            if date_range and 'start' in date_range:
                start_year_int = date_range['start']
            else:
                start_year_int = start_year  # Use the computed start_year

            if not keywords_with_synonyms:
                update_search_progress(current_user.id, query, "error: No valid keywords found")
                response = make_response(render_template('search.html', 
                    current_year=current_year,  # Pass current_year
                    error="No valid keywords found", prompts=prompts, prompt_id=prompt_id, 
                    prompt_text=selected_prompt_text, sources=[], total_results={}, total_pages={}, page=page, per_page=per_page, 
                    username=current_user.email, has_prompt=bool(selected_prompt_text), prompt_params={}, 
                    summary_result_count=20, search_older=search_older, start_year=start_year, sort_by=sort_by, 
                    pubmed_results=[], pubmed_fallback_results=[], sources_selected=sources_selected, combined_summary='', result_limit=result_limit))
                response.headers['X-Content-Type-Options'] = 'nosniff'
                return response

            # ... (rest of the search logic remains unchanged until rendering)

            response = make_response(render_template(
                'search.html', 
                current_year=current_year,  # Pass current_year
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
                current_year=current_year,  # Pass current_year
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
        current_year=current_year,  # Pass current_year
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
        start_year=None,  # Initial render shows "Any", logic applied on submit
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