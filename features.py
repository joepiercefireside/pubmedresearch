import os
from flask import render_template, request, redirect, url_for, flash, jsonify, make_response, session
from flask_login import login_required, current_user
import psycopg2
import json
import hashlib
import time
from datetime import datetime, timedelta
from apscheduler.triggers.cron import CronTrigger
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content, HtmlContent
from core import app, logger, update_search_progress, query_grok_api, scheduler, sg, generate_embedding, get_db_connection
from search_utils import save_search_results, get_search_results, rank_results, markdown_to_html
from prompt_utils import parse_prompt
from auth import validate_user_email
from utils import extract_keywords_and_date, PubMedSearchHandler, GoogleScholarSearchHandler, SemanticScholarSearchHandler
import numpy as np
from scipy.spatial.distance import cosine

def save_search_history(user_id, query, prompt_text, sources, results, search_id=None):
    result_ids = save_search_results(user_id, query, results)
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO search_history (id, user_id, query, prompt_text, sources, result_ids, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (search_id or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16], user_id, query, prompt_text, json.dumps(sources), json.dumps(result_ids), time.time())
        )
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error saving search history: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()
    logger.info(f"Saved search history for user={user_id}, query={query}, search_id={search_id}")
    return result_ids

def clean_invalid_search_history(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, result_ids FROM search_history WHERE user_id = %s", (user_id,))
        for row in cur.fetchall():
            search_id, result_ids = row
            try:
                json.loads(result_ids) if result_ids else []
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in result_ids for search_id={search_id}, user_id={user_id}")
                cur.execute("UPDATE search_history SET result_ids = '[]' WHERE id = %s AND user_id = %s", (search_id, user_id))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error cleaning search history: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def get_search_history(user_id, retention_hours=24):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cutoff_time = time.time() - (retention_hours * 3600)
        cur.execute("""
            SELECT id, query, prompt_text, sources, result_ids, timestamp 
            FROM search_history 
            WHERE user_id = %s AND timestamp > %s 
            ORDER BY timestamp DESC
        """, (user_id, cutoff_time))
        results = []
        for row in cur.fetchall():
            try:
                result_ids = json.loads(row[4]) if row[4] else []
                results.append({
                    'id': row[0],
                    'query': row[1],
                    'prompt_text': row[2],
                    'sources': json.loads(row[3]) if row[3] and isinstance(row[3], str) else [],
                    'result_ids': result_ids,
                    'timestamp': row[5]
                })
            except json.JSONDecodeError:
                logger.warning(f"Skipping search_id={row[0]} due to invalid result_ids JSON")
                continue
        logger.debug(f"Retrieved {len(results)} search history entries for user={user_id}")
        return results
    except psycopg2.Error as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def delete_search_history(user_id, period):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        periods = {
            'weekly': 7 * 86400,
            'monthly': 31 * 86400,
            'annually': 365 * 86400
        }
        cutoff_time = time.time() - periods.get(period, 7 * 86400)
        cur.execute("DELETE FROM search_history WHERE user_id = %s AND timestamp < %s", (user_id, cutoff_time))
        conn.commit()
        logger.info(f"Deleted search history for user={user_id}, period={period}")
    except psycopg2.Error as e:
        logger.error(f"Error deleting search history: {str(e)}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def delete_single_search(user_id, search_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM search_history WHERE user_id = %s AND id = %s", (user_id, search_id))
        if cur.rowcount == 0:
            logger.warning(f"No search found with id={search_id} for user={user_id}")
            return False
        conn.commit()
        logger.info(f"Deleted search id={search_id} for user={user_id}")
        return True
    except psycopg2.Error as e:
        logger.error(f"Error deleting single search: {str(e)}")
        conn.rollback()
        return False
    finally:
        cur.close()
        conn.close()

def save_chat_message(user_id, session_id, message, is_user, search_ids=None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO chat_history (user_id, session_id, message, is_user, timestamp, search_id) VALUES (%s, %s, %s, %s, %s, %s)",
                    (user_id, session_id, message, is_user, time.time(), json.dumps(search_ids) if search_ids else None))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error saving chat message: {str(e)}")
    finally:
        cur.close()
        conn.close()

def get_chat_history(user_id, session_id, search_ids=None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if search_ids:
            cur.execute("SELECT message, is_user, timestamp, search_id FROM chat_history WHERE user_id = %s AND session_id = %s AND search_id = %s ORDER BY timestamp ASC",
                        (user_id, session_id, json.dumps(search_ids)))
        else:
            cur.execute("SELECT message, is_user, timestamp, search_id FROM chat_history WHERE user_id = %s AND session_id = %s ORDER BY timestamp ASC",
                        (user_id, session_id))
        messages = [{'message': row[0], 'is_user': row[1], 'timestamp': row[2], 'search_ids': json.loads(row[3]) if row[3] else None} for row in cur.fetchall()]
        return messages
    except (psycopg2.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def get_user_settings(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT chat_memory_retention_hours FROM user_settings WHERE user_id = %s", (user_id,))
        result = cur.fetchone()
        return result[0] if result else 24
    except psycopg2.Error as e:
        logger.error(f"Error retrieving user settings: {str(e)}")
        return 24
    finally:
        cur.close()
        conn.close()

def update_user_settings(user_id, retention_hours):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO user_settings (user_id, chat_memory_retention_hours) VALUES (%s, %s)
            ON CONFLICT (user_id) DO UPDATE SET chat_memory_retention_hours = %s
        """, (user_id, retention_hours, retention_hours))
        conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Error updating user settings: {str(e)}")
    finally:
        cur.close()
        conn.close()

def run_notification_rule(rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources, test_mode=False):
    logger.info(f"Running notification rule {rule_id} ({rule_name}) for user {user_id}, keywords: {keywords}, timeframe: {timeframe}, sources: {sources}, test_mode={test_mode}, recipient: {user_email}")
    if not validate_user_email(user_email):
        raise ValueError(f"Invalid recipient email address: {user_email}")

    query = keywords
    today = datetime.now()
    timeframe_days = {
        'daily': 1,
        'weekly': 7,
        'monthly': 31,
        'annually': 365
    }
    days = timeframe_days[timeframe]
    start_date = (today - timedelta(days=days)).strftime('%Y/%m/%d')
    end_date = today.strftime('%Y/%m/%d')
    date_range = f"{start_date}:{end_date}"

    prompt_params = parse_prompt(prompt_text) or {}
    result_limit = 50
    summary_result_count = prompt_params.get('summary_result_count', 20)

    search_handlers = {
        'pubmed': PubMedSearchHandler(),
        'googlescholar': GoogleScholarSearchHandler(),
        'semanticscholar': SemanticScholarSearchHandler()
    }

    try:
        update_search_progress(user_id, query, f"Searching {', '.join(sources)} for notification rule")
        keywords_with_synonyms, _, start_year_int = extract_keywords_and_date(query)
        results = []
        for source_id in sources:
            if source_id not in search_handlers:
                logger.warning(f"Unknown source in notification: {source_id}")
                continue
            handler = search_handlers[source_id]
            primary_results, _ = handler.search(query, keywords_with_synonyms, date_range, start_year_int, result_limit=result_limit)
            if primary_results:
                ranked_results = rank_results(query, primary_results, prompt_params)
                results.extend([dict(r, source_id=source_id) for r in ranked_results[:summary_result_count]])

        if results:
            save_search_results(user_id, query, results)

        logger.info(f"Notification rule {rule_id} retrieved {len(results)} results")
        if not results:
            content = "No new results found for this rule."
            try:
                html_content = render_template('email_notification.html', query=query, rule_name=rule_name, user_email=user_email, results=[], summary="", sources=sources, email_format=email_format, summary_result_count=summary_result_count)
                plain_content = render_template('email_notification.txt', query=query, rule_name=rule_name, user_email=user_email, results=[], summary="", sources=sources, email_format=email_format, summary_result_count=summary_result_count)
                logger.debug(f"Rendered no-results email: HTML length={len(html_content)}, Plain text length={len(plain_content)}")
            except Exception as e:
                logger.error(f"Error rendering no-results email templates: {str(e)}")
                raise
            if not sg:
                logger.error("SendGrid client not initialized: SENDGRID_API_KEY missing or invalid")
                raise Exception("SendGrid API key not configured.")
            try:
                message = Mail(
                    from_email=Email("noreply@firesidetechnologies.com"),
                    to_emails=To(user_email),
                    subject=f"AI Research Agent {'Test ' if test_mode else ''}Notification: {rule_name}",
                    plain_text_content=plain_content,
                    html_content=HtmlContent(html_content)
                )
                message.mime_type = 'multipart/alternative'
                logger.debug(f"Sending no-results email with MIME type: {message.mime_type}")
                response = sg.send(message)
                response_headers = {k: v for k, v in response.headers.items()}
                logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}, message_id={response_headers.get('X-Message-Id', 'Not provided')}")
            except Exception as e:
                logger.error(f"Failed to send notification email for rule {rule_id}: {str(e)}")
                if "403" in str(e):
                    raise Exception("SendGrid API authentication failed: Invalid API key or unverified sender email (noreply@firesidetechnologies.com)")
                raise

            if test_mode:
                return {
                    "results": [],
                    "summary": "No new results found.",
                    "email_content": plain_content,
                    "html_content": html_content,
                    "status": "success",
                    "email_sent": True,
                    "message_id": response_headers.get('X-Message-Id', 'Not provided')
                }
            return

        context = "\n".join([
            f"Title: {r['title']}\n"
            f"Abstract: {r.get('abstract', '')}\n"
            f"Authors: {r.get('authors', 'N/A')}\n"
            f"Date: {r.get('publication_date', 'N/A')}\n"
            f"URL: {r.get('url', 'N/A')}"
            for r in results[:summary_result_count]
        ])
        summary_prompt = prompt_text or """
Summarize the provided research articles in a concise manner using Markdown. Use:
- **Bold** for key terms
- Bullet points for main findings
- [Hyperlinks](URL) for article references
- Separate paragraphs for each article
"""
        output = query_grok_api(summary_prompt, context)

        if email_format == "list":
            content = "\n".join([
                f"- [{r['title']}]({r.get('url', 'N/A')})"
                for r in results[:summary_result_count]
            ])
        elif email_format == "detailed":
            content = "\n".join([
                f"**Title**: [{r['title']}]({r.get('url', 'N/A')})\n"
                f"**Authors**: {r.get('authors', 'N/A')}\n"
                f"**Journal**: {r.get('journal', 'N/A')}\n"
                f"**Date**: {r.get('publication_date', 'N/A')}\n"
                f"**Abstract**: {r.get('abstract', '') or 'No abstract'}\n"
                for r in results[:summary_result_count]
            ])
        else:
            content = output

        try:
            html_content = render_template('email_notification.html', query=query, rule_name=rule_name, user_email=user_email, results=results, summary=output, sources=sources, email_format=email_format, summary_result_count=summary_result_count)
            plain_content = render_template('email_notification.txt', query=query, rule_name=rule_name, user_email=user_email, results=results, summary=output, sources=sources, email_format=email_format, summary_result_count=summary_result_count)
            logger.debug(f"Rendered results email: HTML length={len(html_content)}, Plain text length={len(plain_content)}")
        except Exception as e:
            logger.error(f"Error rendering results email templates: {str(e)}")
            raise
        if not sg:
            logger.error("SendGrid client not initialized: SENDGRID_API_KEY missing or invalid")
            raise Exception("SendGrid API key not configured.")
        try:
            message = Mail(
                from_email=Email("noreply@firesidetechnologies.com"),
                to_emails=To(user_email),
                subject=f"AI Research Agent {'Test ' if test_mode else ''}Notification: {rule_name}",
                plain_text_content=plain_content,
                html_content=HtmlContent(html_content)
            )
            message.mime_type = 'multipart/alternative'
            logger.debug(f"Sending results email with MIME type: {message.mime_type}")
            response = sg.send(message)
            response_headers = {k: v for k, v in response.headers.items()}
            logger.info(f"Email sent for rule {rule_id}, status: {response.status_code}, message_id={response_headers.get('X-Message-Id', 'Not provided')}")
        except Exception as e:
            logger.error(f"Failed to send notification email for rule {rule_id}: {str(e)}")
            if "403" in str(e):
                raise Exception("SendGrid API authentication failed: Invalid API key or unverified sender email (noreply@firesidetechnologies.com)")
            raise

        if test_mode:
            return {
                "results": results,
                "summary": output,
                "email_content": plain_content,
                "html_content": html_content,
                "status": "success",
                "email_sent": True,
                "message_id": response_headers.get('X-Message-Id', 'Not provided')
            }

    except Exception as e:
        logger.error(f"Error running notification rule {rule_id}: {str(e)}")
        if test_mode:
            try:
                if not sg:
                    logger.error("SendGrid client not initialized: SENDGRID_API_KEY missing or invalid")
                    raise Exception("SendGrid API key not configured.")
                html_content = render_template('email_notification.html', query=query, rule_name=rule_name, user_email=user_email, results=[], summary="", sources=sources, email_format=email_format, summary_result_count=summary_result_count, error=str(e))
                plain_content = render_template('email_notification.txt', query=query, rule_name=rule_name, user_email=user_email, results=[], summary="", sources=sources, email_format=email_format, summary_result_count=summary_result_count, error=str(e))
                logger.debug(f"Rendered error email: HTML length={len(html_content)}, Plain text length={len(plain_content)}")
                message = Mail(
                    from_email=Email("noreply@firesidetechnologies.com"),
                    to_emails=To(user_email),
                    subject=f"AI Research Agent Test Notification Failed: {rule_name}",
                    plain_text_content=plain_content,
                    html_content=HtmlContent(html_content)
                )
                message.mime_type = 'multipart/alternative'
                logger.debug(f"Sending error email with MIME type: {message.mime_type}")
                response = sg.send(message)
                response_headers = {k: v for k, v in response.headers.items()}
                logger.info(f"Error email sent for rule {rule_id}, status: {response.status_code}, message_id={response_headers.get('X-Message-Id', 'Not provided')}")
                email_sent = True
            except Exception as email_e:
                logger.error(f"Failed to send error email for rule {rule_id}: {str(email_e)}")
                email_sent = False
            error_message = str(e) if "SendGrid" in str(e) else f"Error testing notification: {str(e)}"
            return {
                "results": [],
                "summary": "",
                "email_content": error_message,
                "html_content": html_content if email_sent else "",
                "status": "error",
                "email_sent": email_sent,
                "message_id": response_headers.get('X-Message-Id', 'Not provided') if email_sent else None
            }
        raise

def schedule_notification_rules():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT n.id, n.user_id, n.rule_name, n.keywords, n.timeframe, n.prompt_text, n.email_format, u.email, n.sources "
            "FROM notifications n JOIN users u ON n.user_id = u.id"
        )
        rules = cur.fetchall()
        for rule in rules:
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources = rule
            try:
                sources = json.loads(sources) if sources and isinstance(sources, str) else ['pubmed']
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Invalid sources JSON for notification rule {rule_id}: {sources}")
                sources = ['pubmed']
            cron_trigger = {
                'daily': CronTrigger(hour=8, minute=0),
                'weekly': CronTrigger(day_of_week='mon', hour=8, minute=0),
                'monthly': CronTrigger(day=1, hour=8, minute=0),
                'annually': CronTrigger(month=1, day=1, hour=8, minute=0)
            }[timeframe]
            scheduler.add_job(
                run_notification_rule,
                trigger=cron_trigger,
                args=[rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, user_email, sources],
                id=f"notification_{rule_id}",
                replace_existing=True
            )
        logger.info(f"Scheduled {len(rules)} notification rules")
    except psycopg2.Error as e:
        logger.error(f"Error scheduling notification rules: {str(e)}")
    finally:
        cur.close()
        conn.close()

@app.errorhandler(TypeError)
def handle_type_error(e):
    logger.error(f"TypeError in application: {str(e)}")
    flash("Invalid request parameters", "error")
    return redirect(url_for('search'))

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    # Clean invalid search history entries
    clean_invalid_search_history(current_user.id)
    
    # Generate new session ID to clear chat history
    session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()
    session['chat_session_id'] = session_id
    
    retention_hours = get_user_settings(current_user.id)
    search_ids = request.args.getlist('search_id') or session.get('selected_search_ids', [])
    # Filter invalid search_ids
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if search_ids:
            cur.execute("""
                SELECT id FROM search_history 
                WHERE user_id = %s AND id = ANY(%s) 
            """, (current_user.id, search_ids))
            valid_search_ids = [row[0] for row in cur.fetchall()]
            if valid_search_ids != search_ids:
                session['selected_search_ids'] = valid_search_ids
                session.modified = True
                logger.info(f"Filtered search_ids for user={current_user.id}: {valid_search_ids}")
        else:
            session.pop('selected_search_ids', None)
    except psycopg2.Error as e:
        logger.error(f"Error validating search_ids: {str(e)}")
        session.pop('selected_search_ids', None)
    finally:
        cur.close()
        conn.close()

    chat_history = get_chat_history(current_user.id, session_id, search_ids if search_ids else None)
    chat_history = [
        {
            'message': msg['message'],
            'is_user': msg['is_user'],
            'timestamp': msg['timestamp'],
            'html_message': markdown_to_html(msg['message']) if not msg['is_user'] else None
        } for msg in chat_history
    ]
    search_history = get_search_history(current_user.id, retention_hours)
    
    if request.method == 'POST' and 'retention_hours' in request.form:
        new_retention = request.form.get('retention_hours')
        if new_retention:
            try:
                retention_hours = int(new_retention)
                if retention_hours < 1 or retention_hours > 720:
                    flash("Retention hours must be between 1 and 720.", "error")
                else:
                    update_user_settings(current_user.id, retention_hours)
                    flash("Search retention period updated.", "success")
                    return redirect(url_for('chat', search_id=search_ids))
            except ValueError:
                flash("Invalid retention hours.", "error")
    
    response = make_response(render_template('chat.html', chat_history=chat_history, search_history=search_history, username=current_user.email, retention_hours=retention_hours, search_ids=search_ids))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/chat/select_searches', methods=['POST'])
@login_required
def select_searches():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    data = request.get_json()
    logger.debug(f"Received select_searches request for user={current_user.id}: data={data}")
    selected_searches = data.get('selected_searches', [])
    
    if not selected_searches:
        logger.warning(f"No searches selected for user={current_user.id}")
        return jsonify({'status': 'error', 'message': 'Please select at least one search to chat about'}), 400
    
    # Validate selected searches
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT id FROM search_history 
            WHERE user_id = %s AND id = ANY(%s) 
        """, (current_user.id, selected_searches))
        valid_search_ids = [row[0] for row in cur.fetchall()]
        if not valid_search_ids:
            logger.warning(f"No valid searches found for user={current_user.id}, selected_searches={selected_searches}")
            return jsonify({'status': 'error', 'message': 'No valid searches found for selection'}), 400
        session['selected_search_ids'] = valid_search_ids
        session.modified = True
        logger.info(f"Selected search IDs {valid_search_ids} for user {current_user.id}, session={session.get('chat_session_id')}")
        return jsonify({'status': 'success', 'message': 'Searches selected'})
    except psycopg2.Error as e:
        logger.error(f"Error validating selected searches: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Error selecting searches'}), 400
    finally:
        cur.close()
        conn.close()

@app.route('/chat_message', methods=['POST'])
@login_required
def chat_message():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    session_id = session.get('chat_session_id', str(hashlib.sha256(str(time.time()).encode()).hexdigest()))
    data = request.get_json()
    message = data.get('message', '').strip()
    search_ids = session.get('selected_search_ids', [])
    
    logger.debug(f"Chat message request: user={current_user.id}, session_id={session_id}, message={message[:50]}..., search_ids={search_ids}")
    
    if not message:
        logger.warning(f"Empty message received for user={current_user.id}")
        return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400
    
    save_chat_message(current_user.id, session_id, message, True, search_ids)
    
    if not search_ids:
        logger.warning(f"No search IDs selected for user={current_user.id}, session_id={session_id}")
        return jsonify({
            'status': 'error',
            'message': 'Please select at least one search to chat about.'
        }), 400
    
    search_results = []
    for search_id in search_ids:
        results = get_search_results(current_user.id, search_id)
        logger.debug(f"Retrieved {len(results)} results for user={current_user.id}, search_id={search_id}")
        search_results.extend(results)
    
    if not search_results:
        logger.warning(f"No search results found for user={current_user.id}, search_ids={search_ids}")
        return jsonify({
            'status': 'error',
            'message': 'No relevant search results found for the selected searches.'
        }), 400
    
    # Limit to top 20 results
    search_results = search_results[:20]
    context = "\n".join([
        f"Title: {r['title']}\nAbstract: {r.get('abstract', 'N/A')}\nAuthors: {r.get('authors', 'N/A')}\nJournal: {r.get('journal', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nURL: {r.get('url', 'N/A')}"
        for r in search_results
    ])
    
    system_prompt = """
You are an AI research assistant designed to provide responses based solely on the provided search results. Do not use any external knowledge or assumptions. Focus on clarifying or finding specific details within the selected search results. Use Markdown for formatting, including:
- **Bold** for key terms
- Bullet points for main points
- [Hyperlinks](URL) for article references
Respond directly to the user's question, using only the context provided.
"""
    ai_response = query_grok_api(system_prompt + "\nUser: " + message, context)
    save_chat_message(current_user.id, session_id, ai_response, False, search_ids)
    
    logger.info(f"Chat response generated for user={current_user.id}, message={message[:50]}..., response_length={len(ai_response)}")
    return jsonify({
        'status': 'success',
        'message': ai_response,
        'html_message': markdown_to_html(ai_response)
    })

@app.route('/previous_searches', methods=['GET', 'POST'])
@login_required
def previous_searches():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    # Clean invalid search history entries
    clean_invalid_search_history(current_user.id)
    
    if request.method == 'POST':
        if 'delete_period' in request.form:
            period = request.form.get('delete_period')
            delete_search_history(current_user.id, period)
            flash(f"Deleted searches older than {period}", "success")
        elif 'delete_search_id' in request.form:
            search_id = request.form.get('delete_search_id')
            if delete_single_search(current_user.id, search_id):
                flash("Search deleted successfully", "success")
            else:
                flash("Search not found", "error")
        elif 'rerun_search_id' in request.form:
            search_id = request.form.get('rerun_search_id')
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute("SELECT query, prompt_text, sources FROM search_history WHERE id = %s AND user_id = %s",
                            (search_id, current_user.id))
                result = cur.fetchone()
                if result:
                    query, prompt_text, sources = result
                    try:
                        sources = json.loads(sources) if sources and isinstance(sources, str) else []
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid sources JSON for search {search_id}: {sources}")
                        sources = []
                    return redirect(url_for('search', query=query, prompt_text=prompt_text, sources=sources))
                flash("Search not found for re-run", "error")
            except psycopg2.Error as e:
                logger.error(f"Error rerunning search: {str(e)}")
                flash("Error rerunning search", "error")
            finally:
                cur.close()
                conn.close()
    
    retention_hours = get_user_settings(current_user.id)
    searches = get_search_history(current_user.id, retention_hours)
    response = make_response(render_template('previous_searches.html', searches=searches, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, rule_name, keywords, sources, timeframe, prompt_text, email_format, created_at FROM notifications WHERE user_id = %s",
                    (current_user.id,))
        notifications = [
            {
                'id': row[0],
                'rule_name': row[1],
                'keywords': row[2],
                'sources': json.loads(row[3]) if row[3] and isinstance(row[3], str) else [],
                'timeframe': row[4],
                'prompt_text': row[5],
                'email_format': row[6],
                'created_at': row[7]
            } for row in cur.fetchall()
        ]
    except (psycopg2.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error retrieving notifications: {str(e)}")
        notifications = []
    finally:
        cur.close()
        conn.close()

    if request.method == 'POST':
        if 'rule_name' in request.form:
            rule_name = request.form.get('rule_name')
            keywords = request.form.get('keywords')
            sources = request.form.getlist('sources')
            timeframe = request.form.get('timeframe')
            prompt_text = request.form.get('prompt_text')
            email_format = request.form.get('email_format')
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO notifications (user_id, rule_name, keywords, sources, timeframe, prompt_text, email_format, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    (current_user.id, rule_name, keywords, json.dumps(sources), timeframe, prompt_text, email_format, datetime.now())
                )
                conn.commit()
                flash("Notification rule added successfully", "success")
                scheduler.add_job(
                    run_notification_rule,
                    trigger=CronTrigger(hour=8, minute=0) if timeframe == 'daily' else
                            CronTrigger(day_of_week='mon', hour=8, minute=0) if timeframe == 'weekly' else
                            CronTrigger(day=1, hour=8, minute=0) if timeframe == 'monthly' else
                            CronTrigger(month=1, day=1, hour=8, minute=0),
                    args=[cur.lastrowid, current_user.id, rule_name, keywords, timeframe, prompt_text, email_format, current_user.email, sources],
                    id=f"notification_{cur.lastrowid}",
                    replace_existing=True
                )
            except psycopg2.Error as e:
                logger.error(f"Error adding notification rule: {str(e)}")
                conn.rollback()
                flash("Error adding notification rule", "error")
            finally:
                cur.close()
                conn.close()
        elif 'delete_rule_id' in request.form:
            rule_id = request.form.get('delete_rule_id')
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute("DELETE FROM notifications WHERE id = %s AND user_id = %s", (rule_id, current_user.id))
                if cur.rowcount == 0:
                    flash("Notification rule not found", "error")
                else:
                    conn.commit()
                    scheduler.remove_job(f"notification_{rule_id}")
                    flash("Notification rule deleted successfully", "success")
            except psycopg2.Error as e:
                logger.error(f"Error deleting notification rule: {str(e)}")
                conn.rollback()
                flash("Error deleting notification rule", "error")
            finally:
                cur.close()
                conn.close()
        elif 'test_rule_id' in request.form:
            rule_id = request.form.get('test_rule_id')
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute("SELECT rule_name, keywords, timeframe, prompt_text, email_format, sources FROM notifications WHERE id = %s AND user_id = %s",
                            (rule_id, current_user.id))
                result = cur.fetchone()
                if result:
                    rule_name, keywords, timeframe, prompt_text, email_format, sources = result
                    try:
                        sources = json.loads(sources) if sources and isinstance(sources, str) else []
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(f"Invalid sources JSON for notification rule {rule_id}: {sources}")
                        sources = []
                    test_result = run_notification_rule(rule_id, current_user.id, rule_name, keywords, timeframe, prompt_text, email_format, current_user.email, sources, test_mode=True)
                    flash(f"Test email sent. Status: {test_result['status']}. Message ID: {test_result['message_id']}", "success")
                else:
                    flash("Notification rule not found for testing", "error")
            except Exception as e:
                logger.error(f"Error testing notification rule: {str(e)}")
                flash(f"Error testing notification rule: {str(e)}", "error")
            finally:
                cur.close()
                conn.close()

    response = make_response(render_template('notifications.html', notifications=notifications, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notification_edit/<rule_id>', methods=['GET', 'POST'])
@login_required
def notification_edit(rule_id):
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT rule_name, keywords, sources, timeframe, prompt_text, email_format FROM notifications WHERE id = %s AND user_id = %s",
                    (rule_id, current_user.id))
        result = cur.fetchone()
        if not result:
            flash("Notification rule not found", "error")
            return redirect(url_for('notifications'))
        notification = {
            'rule_name': result[0],
            'keywords': result[1],
            'sources': json.loads(result[2]) if result[2] and isinstance(result[2], str) else [],
            'timeframe': result[3],
            'prompt_text': result[4],
            'email_format': result[5]
        }
    except (psycopg2.Error, json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error retrieving notification: {str(e)}")
        flash("Error retrieving notification rule", "error")
        return redirect(url_for('notifications'))
    finally:
        cur.close()
        conn.close()

    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        sources = request.form.getlist('sources')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text')
        email_format = request.form.get('email_format')
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE notifications SET rule_name = %s, keywords = %s, sources = %s, timeframe = %s, prompt_text = %s, email_format = %s WHERE id = %s AND user_id = %s",
                (rule_name, keywords, json.dumps(sources), timeframe, prompt_text, email_format, rule_id, current_user.id)
            )
            if cur.rowcount == 0:
                flash("Notification rule not found", "error")
            else:
                conn.commit()
                flash("Notification rule updated successfully", "success")
                scheduler.remove_job(f"notification_{rule_id}")
                scheduler.add_job(
                    run_notification_rule,
                    trigger=CronTrigger(hour=8, minute=0) if timeframe == 'daily' else
                            CronTrigger(day_of_week='mon', hour=8, minute=0) if timeframe == 'weekly' else
                            CronTrigger(day=1, hour=8, minute=0) if timeframe == 'monthly' else
                            CronTrigger(month=1, day=1, hour=8, minute=0),
                    args=[rule_id, current_user.id, rule_name, keywords, timeframe, prompt_text, email_format, current_user.email, sources],
                    id=f"notification_{rule_id}",
                    replace_existing=True
                )
            return redirect(url_for('notifications'))
        except psycopg2.Error as e:
            logger.error(f"Error updating notification rule: {str(e)}")
            conn.rollback()
            flash("Error updating notification rule", "error")
        finally:
            cur.close()
            conn.close()

    response = make_response(render_template('notification_edit.html', notification=notification, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/prompt', methods=['GET', 'POST'])
@login_required
def prompt():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s",
                    (current_user.id,))
        prompts = [
            {
                'id': row[0],
                'prompt_name': row[1],
                'prompt_text': row[2],
                'created_at': row[3]
            } for row in cur.fetchall()
        ]
    except psycopg2.Error as e:
        logger.error(f"Error retrieving prompts: {str(e)}")
        prompts = []
    finally:
        cur.close()
        conn.close()

    if request.method == 'POST':
        if 'prompt_name' in request.form:
            prompt_name = request.form.get('prompt_name')
            prompt_text = request.form.get('prompt_text')
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO prompts (user_id, prompt_name, prompt_text, created_at) VALUES (%s, %s, %s, %s)",
                    (current_user.id, prompt_name, prompt_text, datetime.now())
                )
                conn.commit()
                flash("Prompt added successfully", "success")
            except psycopg2.Error as e:
                logger.error(f"Error adding prompt: {str(e)}")
                conn.rollback()
                flash("Error adding prompt", "error")
            finally:
                cur.close()
                conn.close()
        elif 'delete_prompt_id' in request.form:
            prompt_id = request.form.get('delete_prompt_id')
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute("DELETE FROM prompts WHERE id = %s AND user_id = %s", (prompt_id, current_user.id))
                if cur.rowcount == 0:
                    flash("Prompt not found", "error")
                else:
                    conn.commit()
                    flash("Prompt deleted successfully", "success")
            except psycopg2.Error as e:
                logger.error(f"Error deleting prompt: {str(e)}")
                conn.rollback()
                flash("Error deleting prompt", "error")
            finally:
                cur.close()
                conn.close()
        return redirect(url_for('prompt'))

    response = make_response(render_template('prompt.html', prompts=prompts, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/prompt_edit/<prompt_id>', methods=['GET', 'POST'])
@login_required
def prompt_edit(prompt_id):
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT prompt_name, prompt_text FROM prompts WHERE id = %s AND user_id = %s",
                    (prompt_id, current_user.id))
        result = cur.fetchone()
        if not result:
            flash("Prompt not found", "error")
            return redirect(url_for('prompt'))
        prompt = {
            'prompt_name': result[0],
            'prompt_text': result[1]
        }
    except psycopg2.Error as e:
        logger.error(f"Error retrieving prompt: {str(e)}")
        flash("Error retrieving prompt", "error")
        return redirect(url_for('prompt'))
    finally:
        cur.close()
        conn.close()

    if request.method == 'POST':
        prompt_name = request.form.get('prompt_name')
        prompt_text = request.form.get('prompt_text')
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE prompts SET prompt_name = %s, prompt_text = %s WHERE id = %s AND user_id = %s",
                (prompt_name, prompt_text, prompt_id, current_user.id)
            )
            if cur.rowcount == 0:
                flash("Prompt not found", "error")
            else:
                conn.commit()
                flash("Prompt updated successfully", "success")
            return redirect(url_for('prompt'))
        except psycopg2.Error as e:
            logger.error(f"Error updating prompt: {str(e)}")
            conn.rollback()
            flash("Error updating prompt", "error")
        finally:
            cur.close()
            conn.close()

    response = make_response(render_template('prompt_edit.html', prompt=prompt, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response