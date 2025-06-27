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

def get_search_history(user_id, days=7):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cutoff_time = time.time() - (days * 86400)
        cur.execute("SELECT id, query, prompt_text, sources, result_ids, timestamp FROM search_history WHERE user_id = %s AND timestamp > %s ORDER BY timestamp DESC",
                    (user_id, cutoff_time))
        results = [
            {'id': row[0], 'query': row[1], 'prompt_text': row[2], 'sources': json.loads(row[3]), 'result_ids': json.loads(row[4]), 'timestamp': row[5]}
            for row in cur.fetchall()
        ]
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

def get_chat_history(user_id, session_id, retention_hours, search_ids=None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cutoff_time = time.time() - (retention_hours * 3600)
        if search_ids:
            cur.execute("SELECT message, is_user, timestamp, search_id FROM chat_history WHERE user_id = %s AND session_id = %s AND timestamp > %s AND search_id = %s ORDER BY timestamp ASC",
                        (user_id, session_id, cutoff_time, json.dumps(search_ids)))
        else:
            cur.execute("SELECT message, is_user, timestamp, search_id FROM chat_history WHERE user_id = %s AND session_id = %s AND timestamp > %s ORDER BY timestamp ASC",
                        (user_id, session_id, cutoff_time))
        messages = [{'message': row[0], 'is_user': row[1], 'timestamp': row[2], 'search_ids': json.loads(row[3]) if row[3] else None} for row in cur.fetchall()]
        return messages
    except psycopg2.Error as e:
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
            sources = json.loads(sources) if sources else ['pubmed']
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

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if not current_user.is_authenticated:
        response = make_response(redirect(url_for('login')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    session_id = session.get('chat_session_id', str(hashlib.sha256(str(time.time()).encode()).hexdigest()))
    session['chat_session_id'] = session_id
    
    retention_hours = get_user_settings(current_user.id)
    search_ids = request.args.getlist('search_id') or session.get('selected_search_ids', [])
    if search_ids:
        session['selected_search_ids'] = search_ids
    chat_history = get_chat_history(current_user.id, session_id, retention_hours, search_ids if search_ids else None)
    chat_history = [
        {
            'message': msg['message'],
            'is_user': msg['is_user'],
            'timestamp': msg['timestamp'],
            'html_message': markdown_to_html(msg['message']) if not msg['is_user'] else None
        } for msg in chat_history
    ]
    search_history = get_search_history(current_user.id, retention_hours / 24)
    
    if request.method == 'POST' and 'retention_hours' in request.form:
        new_retention = request.form.get('retention_hours')
        if new_retention:
            try:
                retention_hours = int(new_retention)
                if retention_hours < 1 or retention_hours > 720:
                    flash("Retention hours must be between 1 and 720.", "error")
                else:
                    update_user_settings(current_user.id, retention_hours)
                    flash("Chat memory retention updated.", "success")
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
    selected_searches = data.get('selected_searches', [])
    
    if not selected_searches:
        return jsonify({'status': 'error', 'message': 'At least one search must be selected'}), 400
    
    session['selected_search_ids'] = selected_searches
    logger.info(f"Selected search IDs {selected_searches} for user {current_user.id}")
    return jsonify({'status': 'success', 'message': 'Searches selected'})

@app.route('/chat_message', methods=['POST'])
@login_required
def chat_message():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    session_id = session.get('chat_session_id', str(hashlib.sha256(str(time.time()).encode()).hexdigest()))
    data = request.get_json()
    message = data.get('message', '').strip()
    search_ids = session.get('selected_search_ids', [])
    
    if not message:
        return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400
    
    save_chat_message(current_user.id, session_id, message, True, search_ids)
    
    retention_hours = get_user_settings(current_user.id)
    chat_history = get_chat_history(current_user.id, session_id, retention_hours, search_ids if search_ids else None)
    context = "\n".join([msg['message'] for msg in chat_history if not msg['is_user']])
    
    if search_ids:
        search_results = []
        for search_id in search_ids:
            results = get_search_results(current_user.id, search_id)
            search_results.extend(results)
        search_context = "\n".join([
            f"Title: {r['title']}\nAbstract: {r.get('abstract', 'N/A')}\nURL: {r.get('url', 'N/A')}"
            for r in search_results
        ])
        context += "\n\nSearch Results Context:\n" + search_context
    
    system_prompt = """
You are an AI research assistant designed to provide unique, conversational responses based on the provided context and search results. Avoid generic summaries and instead offer insightful, engaging answers tailored to the user's question. Use Markdown for formatting when appropriate.
"""
    ai_response = query_grok_api(system_prompt + "\nUser: " + message, context)
    save_chat_message(current_user.id, session_id, ai_response, False, search_ids)
    
    return jsonify({
        'status': 'success',
        'message': ai_response,
        'html_message': markdown_to_html(ai_response)
    })