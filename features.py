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
from core import app, logger, update_search_progress, query_grok_api, scheduler, sg, generate_embedding
from search import save_search_results, get_search_results, rank_results
from prompt_utils import parse_prompt
from auth import validate_user_email
from utils import extract_keywords_and_date, PubMedSearchHandler, GoogleScholarSearchHandler, SemanticScholarSearchHandler
import sqlite3
import numpy as np
from scipy.spatial.distance import cosine

def save_search_history(user_id, query, prompt_text, sources, results):
    result_ids = save_search_results(user_id, query, results)
    
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO search_history (user_id, query, prompt_text, sources, result_ids, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, query, prompt_text, json.dumps(sources), json.dumps(result_ids), time.time())
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving search history: {str(e)}")
        conn.rollback()
    finally:
        c.close()
        conn.close()
    logger.info(f"Saved search history for user={user_id}, query={query}")
    return result_ids

def get_search_history(user_id, days=7):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        cutoff_time = time.time() - (days * 86400)
        c.execute("SELECT id, query, prompt_text, sources, result_ids, timestamp FROM search_history WHERE user_id = ? AND timestamp > ? ORDER BY timestamp DESC",
                  (user_id, cutoff_time))
        results = [
            {'id': row[0], 'query': row[1], 'prompt_text': row[2], 'sources': json.loads(row[3]), 'result_ids': json.loads(row[4]), 'timestamp': row[5]}
            for row in c.fetchall()
        ]
        return results
    except sqlite3.Error as e:
        logger.error(f"Error retrieving search history: {str(e)}")
        return []
    finally:
        c.close()
        conn.close()

def delete_search_history(user_id, period):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        periods = {
            'weekly': 7 * 86400,
            'monthly': 31 * 86400,
            'annually': 365 * 86400
        }
        cutoff_time = time.time() - periods.get(period, 7 * 86400)
        c.execute("DELETE FROM search_history WHERE user_id = ? AND timestamp < ?", (user_id, cutoff_time))
        conn.commit()
        logger.info(f"Deleted search history for user={user_id}, period={period}")
    except sqlite3.Error as e:
        logger.error(f"Error deleting search history: {str(e)}")
        conn.rollback()
    finally:
        c.close()
        conn.close()

def save_chat_message(user_id, session_id, message, is_user):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO chat_history (user_id, session_id, message, is_user, timestamp) VALUES (?, ?, ?, ?, ?)",
                  (user_id, session_id, message, is_user, time.time()))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error saving chat message: {str(e)}")
    finally:
        c.close()
        conn.close()

def get_chat_history(user_id, session_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        cutoff_time = time.time() - (retention_hours * 3600)
        c.execute("SELECT message, is_user, timestamp FROM chat_history WHERE user_id = ? AND session_id = ? AND timestamp > ? ORDER BY timestamp ASC",
                  (user_id, session_id, cutoff_time))
        messages = [{'message': row[0], 'is_user': row[1], 'timestamp': row[2]} for row in c.fetchall()]
        return messages
    except sqlite3.Error as e:
        logger.error(f"Error retrieving chat history: {str(e)}")
        return []
    finally:
        c.close()
        conn.close()

def get_user_settings(user_id):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("SELECT chat_memory_retention_hours FROM user_settings WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return result[0] if result else 24
    except sqlite3.Error as e:
        logger.error(f"Error retrieving user settings: {str(e)}")
        return 24
    finally:
        c.close()
        conn.close()

def update_user_settings(user_id, retention_hours):
    conn = sqlite3.connect('search_progress.db')
    c = conn.cursor()
    try:
        c.execute("INSERT OR REPLACE INTO user_settings (user_id, chat_memory_retention_hours) VALUES (?, ?)",
                  (user_id, retention_hours))
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Error updating user settings: {str(e)}")
    finally:
        c.close()
        conn.close()

def get_db_connection():
    conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
    return conn

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
    result_limit = 50  # Match search page default
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
                message.mime_type = 'multipart/alternative'  # Ensure HTML is prioritized
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
        summary_prompt = prompt_text or "Summarize the provided research articles in a concise manner, using Markdown for formatting with hyperlinks, **bold** text for key terms, and bullet points for lists."
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
        else:  # summary
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
            message.mime_type = 'multipart/alternative'  # Ensure HTML is prioritized
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
                message.mime_type = 'multipart/alternative'  # Ensure HTML is prioritized
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
    scheduler.remove_all_jobs()
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
    except Exception as e:
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
    chat_history = get_chat_history(current_user.id, session_id, retention_hours)
    
    if request.method == 'POST':
        user_message = request.form.get('message', '')
        new_retention = request.form.get('retention_hours')
        
        if new_retention:
            try:
                retention_hours = int(new_retention)
                if retention_hours < 1 or retention_hours > 720:
                    flash("Retention hours must be between 1 and 720.", "error")
                else:
                    update_user_settings(current_user.id, retention_hours)
                    flash("Chat memory retention updated.", "success")
            except ValueError:
                flash("Invalid retention hours.", "error")
        
        if user_message:
            save_chat_message(current_user.id, session_id, user_message, True)
            
            search_results = get_search_results(current_user.id, session.get('latest_query', ''))
            query = session.get('latest_query', '')
            context = "\n".join([f"Source: {r['source_id']}\nTitle: {r['title']}\nAbstract: {r.get('abstract', '')}\nAuthors: {r.get('authors', 'N/A')}\nDate: {r.get('publication_date', 'N/A')}\nURL: {r.get('url', 'N/A')}" for r in search_results[:5]])
            
            system_prompt = session.get('latest_prompt_text', "Answer the user's query based on the provided search results and chat history in a clear, concise, and accurate manner, using Markdown for formatting. Include hyperlinks, bold text for key terms, and bullet points for lists where applicable.")
            
            history_context = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['message']}" for msg in chat_history[-5:]])
            full_context = f"Search Query: {query}\n\nSearch Results:\n{context}\n\nChat History:\n{history_context}\n\nUser Query: {user_message}"
            
            try:
                response = query_grok_api(system_prompt, full_context)
                save_chat_message(current_user.id, session_id, response, False)
                chat_history.append({'message': user_message, 'is_user': True, 'timestamp': time.time()})
                chat_history.append({'message': response, 'is_user': False, 'timestamp': time.time()})
            except Exception as e:
                flash(f"Error generating chat response: {str(e)}", "error")
        
        response = make_response(render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    
    response = make_response(render_template('chat.html', chat_history=chat_history, username=current_user.email, retention_hours=retention_hours))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/chat_message', methods=['POST'])
@login_required
def chat_message():
    if not current_user.is_authenticated:
        return jsonify({'status': 'error', 'message': 'User not authenticated'}), 401
    
    session_id = session.get('chat_session_id', str(hashlib.sha256(str(time.time()).encode()).hexdigest()))
    session['chat_session_id'] = session_id
    
    user_message = request.form.get('message', '')
    if not user_message:
        return jsonify({'status': 'error', 'message': 'Message cannot be empty'}), 400
    
    try:
        save_chat_message(current_user.id, session_id, user_message, True)
        
        retention_hours = get_user_settings(current_user.id)
        chat_history = get_chat_history(current_user.id, session_id, retention_hours)
        
        query = session.get('latest_query', '')
        search_results = get_search_results(current_user.id, query)
        
        query_embedding = generate_embedding(user_message)
        if query_embedding is None:
            logger.error("Failed to generate embedding for user query")
            return jsonify({'status': 'error', 'message': 'Failed to process query'}), 500
        
        ranked_results = []
        for result in search_results:
            text = f"{result['title']} {result.get('abstract', '')}"
            result_embedding = generate_embedding(text)
            if result_embedding is None:
                continue
            similarity = 1 - cosine(query_embedding, result_embedding)
            ranked_results.append((result, similarity))
        
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        top_results = [r[0] for r in ranked_results[:5]]
        
        context = "\n".join([f"**Source**: {r['source_id']}\n**Title**: [{r['title']}]({r.get('url', 'N/A')})\n**Abstract**: {r.get('abstract', '')}\n**Authors**: {r.get('authors', 'N/A')}\n**Date**: {r.get('publication_date', 'N/A')}" for r in top_results])
        
        system_prompt = session.get('latest_prompt_text', "Answer the user's query based on the provided search results and chat history in a clear, concise, and accurate manner, using Markdown for formatting. Include hyperlinks, bold text for key terms, and bullet points for lists where applicable. Focus on results most relevant to the user's query.")
        
        history_context = "\n".join([f"{'User' if msg['is_user'] else 'Assistant'}: {msg['message']}" for msg in chat_history[-5:]])
        full_context = f"Search Query: {query}\n\nSearch Results:\n{context}\n\nChat History:\n{history_context}\n\nUser Query: {user_message}"
        
        response = query_grok_api(system_prompt, full_context)
        if "summary" in user_message.lower() and "top" in user_message.lower():
            formatted_response = ""
            for source_id in ['pubmed', 'googlescholar', 'semanticscholar']:
                source_results = [r for r in top_results if r['source_id'] == source_id][:3]
                if source_results:
                    context = "\n".join([f"**Title**: [{r['title']}]({r.get('url', 'N/A')})\n**Abstract**: {r.get('abstract', '')}" for r in source_results])
                    summary_prompt = f"Summarize the abstracts of the following {source_id} articles in simple terms. Provide one paragraph per article, up to 3 paragraphs, separated by a blank line. Use Markdown with hyperlinks, **bold** text for key terms, and bullet points for lists where applicable."
                    summary = query_grok_api(summary_prompt, context)
                    formatted_response += f"### {source_id.capitalize()} Summaries\n{summary}\n\n"
            response = formatted_response.strip() or response
        
        save_chat_message(current_user.id, session_id, response, False)
        
        return jsonify({'status': 'success', 'message': response})
    except Exception as e:
        logger.error(f"Error in chat_message: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/previous_searches', methods=['GET'])
@login_required
def previous_searches():
    search_history = get_search_history(current_user.id)
    response = make_response(render_template('previous_searches.html', searches=search_history, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/delete_search_history', methods=['POST'])
@login_required
def delete_search_history_endpoint():
    period = request.form.get('delete_period', 'weekly')
    if period not in ['weekly', 'monthly', 'annually']:
        flash('Invalid deletion period selected.', 'error')
        return redirect(url_for('previous_searches'))
    
    try:
        delete_search_history(current_user.id, period)
        flash(f"Search history older than {period} deleted successfully.", 'success')
    except Exception as e:
        logger.error(f"Error deleting search history: {str(e)}")
        flash(f"Failed to delete search history: {str(e)}", 'error')
    
    return redirect(url_for('previous_searches'))

@app.route('/prompt', methods=['GET', 'POST'])
@login_required
def prompt():
    if request.method == 'POST':
        prompt_name = request.form.get('prompt_name')
        prompt_text = request.form.get('prompt_text')
        if not prompt_name or not prompt_text:
            flash('Prompt name and text cannot be empty.', 'error')
        else:
            conn = get_db_connection()
            cur = conn.cursor()
            try:
                cur.execute('INSERT INTO prompts (user_id, prompt_name, prompt_text) VALUES (%s, %s, %s)', 
                            (current_user.id, prompt_name, prompt_text))
                conn.commit()
                flash('Prompt saved successfully.', 'success')
            except Exception as e:
                logger.error(f"Failed to save prompt: {str(e)}")
                conn.rollback()
                flash(f'Failed to save prompt: {str(e)}', 'error')
            finally:
                cur.close()
                conn.close()
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text, created_at FROM prompts WHERE user_id = %s ORDER BY created_at DESC', 
                    (current_user.id,))
        prompts = [{'id': str(p[0]), 'prompt_name': p[1], 'prompt_text': p[2], 'created_at': p[3]} for p in cur.fetchall()]
    except Exception as e:
        logger.error(f"Error loading prompts: {str(e)}")
        prompts = []
    finally:
        cur.close()
        conn.close()
    response = make_response(render_template('prompt.html', prompts=prompts, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/prompt/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, prompt_name, prompt_text FROM prompts WHERE id = %s AND user_id = %s', 
                    (id, current_user.id))
        prompt = cur.fetchone()
        
        if not prompt:
            flash('Prompt not found or you do not have permission to edit it.', 'error')
            response = make_response(redirect(url_for('prompt')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        if request.method == 'POST':
            prompt_name = request.form.get('prompt_name')
            prompt_text = request.form.get('prompt_text')
            if not prompt_name or not prompt_text:
                flash('Prompt name and text cannot be empty.', 'error')
            else:
                try:
                    cur.execute('UPDATE prompts SET prompt_name = %s, prompt_text = %s WHERE id = %s AND user_id = %s', 
                                (prompt_name, prompt_text, id, current_user.id))
                    conn.commit()
                    flash('Prompt updated successfully.', 'success')
                    response = make_response(redirect(url_for('prompt')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
                except Exception as e:
                    logger.error(f"Failed to update prompt: {str(e)}")
                    conn.rollback()
                    flash(f'Failed to update prompt: {str(e)}', 'error')
        
        response = make_response(render_template('prompt_edit.html', prompt={'id': prompt[0], 'prompt_name': prompt[1], 'prompt_text': prompt[2]}, username=current_user.email))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        logger.error(f"Error in edit_prompt: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        response = make_response(redirect(url_for('prompt')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    finally:
        cur.close()
        conn.close()

@app.route('/prompt/delete/<int:id>', methods=['POST'])
@login_required
def delete_prompt(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
        prompt = cur.fetchone()
        
        if not prompt:
            flash('Prompt not found or you do not have permission to delete it.', 'error')
            response = make_response(redirect(url_for('prompt')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        cur.execute('DELETE FROM prompts WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Prompt deleted successfully.', 'success')
    except Exception as e:
        logger.error(f"Error deleting prompt: {str(e)}")
        conn.rollback()
        flash(f'Failed to delete prompt: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    response = make_response(redirect(url_for('prompt')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications', methods=['GET', 'POST'])
@login_required
def notifications():
    conn = get_db_connection()
    cur = conn.cursor()
    
    if request.method == 'POST':
        rule_name = request.form.get('rule_name')
        keywords = request.form.get('keywords')
        timeframe = request.form.get('timeframe')
        prompt_text = request.form.get('prompt_text')
        email_format = request.form.get('email_format')
        sources = request.form.getlist('sources')
        
        if not all([rule_name, keywords, timeframe, sources, email_format]):
            logger.error(f"Missing required fields: rule_name={rule_name}, keywords={keywords}, timeframe={timeframe}, sources={sources}, email_format={email_format}")
            flash('All required fields must be filled.', 'error')
            return redirect(url_for('notifications'))
        elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
            logger.error(f"Invalid timeframe: {timeframe}")
            flash('Invalid timeframe.', 'error')
            return redirect(url_for('notifications'))
        elif email_format not in ['summary', 'list', 'detailed']:
            logger.error(f"Invalid email format: {email_format}")
            flash('Invalid email format.', 'error')
            return redirect(url_for('notifications'))
        
        try:
            cur.execute(
                """
                INSERT INTO notifications (user_id, rule_name, keywords, timeframe, prompt_text, email_format, sources)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (current_user.id, rule_name, keywords, timeframe, prompt_text, email_format, json.dumps(sources))
            )
            conn.commit()
            flash('Notification rule created successfully.', 'success')
            schedule_notification_rules()
        except Exception as e:
            logger.error(f"Error creating notification: {str(e)}")
            conn.rollback()
            flash(f'Failed to create notification rule: {str(e)}', 'error')
    
    try:
        cur.execute(
            "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, created_at, sources "
            "FROM notifications WHERE user_id = %s ORDER BY created_at DESC",
            (current_user.id,)
        )
        notifications = [
            {
                'id': n[0],
                'rule_name': n[1],
                'keywords': n[2],
                'timeframe': n[3],
                'prompt_text': n[4],
                'email_format': n[5],
                'created_at': n[6],
                'sources': json.loads(n[7]) if n[7] else ['pubmed']
            } for n in cur.fetchall()
        ]
    except Exception as e:
        logger.error(f"Error loading notifications: {str(e)}")
        notifications = []
    finally:
        cur.close()
        conn.close()
    response = make_response(render_template('notifications.html', notifications=notifications, username=current_user.email))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications/edit/<int:id>', methods=['GET', 'POST'])
@login_required
def edit_notification(id):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, rule_name, keywords, timeframe, prompt_text, email_format, sources "
            "FROM notifications WHERE id = %s AND user_id = %s",
            (id, current_user.id)
        )
        notification = cur.fetchone()
        
        if not notification:
            logger.error(f"Notification rule {id} not found for user {current_user.id}")
            flash('Notification rule not found or you do not have permission to edit it.', 'error')
            response = make_response(redirect(url_for('notifications')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        if request.method == 'POST':
            rule_name = request.form.get('rule_name')
            keywords = request.form.get('keywords')
            timeframe = request.form.get('timeframe')
            prompt_text = request.form.get('prompt_text')
            email_format = request.form.get('email_format')
            sources = request.form.getlist('sources')
            
            if not all([rule_name, keywords, timeframe, sources, email_format]):
                logger.error(f"Missing required fields: rule_name={rule_name}, keywords={keywords}, timeframe={timeframe}, sources={sources}, email_format={email_format}")
                flash('All required fields must be filled.', 'error')
            elif timeframe not in ['daily', 'weekly', 'monthly', 'annually']:
                logger.error(f"Invalid timeframe: {timeframe}")
                flash('Invalid timeframe selected.', 'error')
            elif email_format not in ['summary', 'list', 'detailed']:
                logger.error(f"Invalid email format: {email_format}")
                flash('Invalid email format selected.', 'error')
            else:
                try:
                    cur.execute(
                        """
                        UPDATE notifications SET rule_name = %s, keywords = %s, timeframe = %s, prompt_text = %s, email_format = %s, sources = %s
                        WHERE id = %s AND user_id = %s
                        """,
                        (rule_name, keywords, timeframe, prompt_text, email_format, json.dumps(sources), id, current_user.id)
                    )
                    conn.commit()
                    flash('Notification rule updated successfully.', 'success')
                    schedule_notification_rules()
                    response = make_response(redirect(url_for('notifications')))
                    response.headers['X-Content-Type-Options'] = 'nosniff'
                    return response
                except Exception as e:
                    logger.error(f"Error updating notification: {str(e)}")
                    conn.rollback()
                    flash(f'Failed to update notification rule: {str(e)}', 'error')
        
        notification_data = {
            'id': notification[0],
            'rule_name': notification[1],
            'keywords': notification[2],
            'timeframe': notification[3],
            'prompt_text': notification[4],
            'email_format': notification[5],
            'sources': json.loads(notification[6]) if notification[6] else ['pubmed']
        }
        response = make_response(render_template('notification_edit.html', notification=notification_data, username=current_user.email))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    except Exception as e:
        logger.error(f"Error in edit_notification: {str(e)}")
        flash('An error occurred. Please try again.', 'error')
        response = make_response(redirect(url_for('notifications')))
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
    finally:
        cur.close()
        conn.close()

@app.route('/notifications/delete/<int:id>', methods=['POST'])
@login_required
def delete_notification(id):
    logger.info(f"Attempting to delete notification rule {id} for user {current_user.id}")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('SELECT id, rule_name FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
        notification = cur.fetchone()
        
        if not notification:
            logger.error(f"Notification rule {id} not found for user {current_user.id}")
            flash('Notification rule not found or you do not have permission to delete it.', 'error')
            response = make_response(redirect(url_for('notifications')))
            response.headers['X-Content-Type-Options'] = 'nosniff'
            return response
        
        logger.info(f"Deleting notification rule {id}: {notification[1]}")
        cur.execute('DELETE FROM notifications WHERE id = %s AND user_id = %s', (id, current_user.id))
        conn.commit()
        flash('Notification rule deleted successfully.', 'success')
        schedule_notification_rules()
    except Exception as e:
        logger.error(f"Error deleting notification rule {id}: {str(e)}")
        conn.rollback()
        flash(f'Failed to delete notification rule: {str(e)}', 'error')
    finally:
        cur.close()
        conn.close()
    
    response = make_response(redirect(url_for('notifications')))
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response

@app.route('/notifications/test/<int:id>', methods=['GET'])
@login_required
def test_notification(id):
    logger.info(f"Testing notification rule {id} for user {current_user.id}")
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, sources "
            "FROM notifications WHERE id = %s AND user_id = %s",
            (id, current_user.id)
        )
        notification = cur.fetchone()
        
        if not notification:
            logger.error(f"Notification rule {id} not found for user {current_user.id}")
            return jsonify({'status': 'error', 'message': 'Notification rule not found or you do not have permission to test it.'}), 404
        
        rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, sources = notification
        sources = json.loads(sources) if sources else ['pubmed']
        logger.debug(f"Test notification {rule_id}: keywords={keywords}, sources={sources}, email={current_user.email}")
        
        result = run_notification_rule(
            rule_id, user_id, rule_name, keywords, timeframe, prompt_text, email_format, current_user.email, sources, test_mode=True
        )
        logger.info(f"Test result for rule {rule_id}: status={result['status']}, email_sent={result.get('email_sent', False)}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error testing notification rule {id}: {str(e)}")
        return jsonify({'status': 'error', 'message': f"Error testing notification: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()