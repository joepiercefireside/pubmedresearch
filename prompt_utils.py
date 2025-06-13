import re
import logging

logger = logging.getLogger(__name__)

def parse_prompt(prompt_text):
    if not prompt_text:
        return {
            'summary_result_count': 20,
            'display_result_count': 20,
            'limit_presentation': False
        }
    
    prompt_text_lower = prompt_text.lower()
    summary_result_count = 20
    if match := re.search(r'(?:top|return|summarize|include|limit\s+to|show\s+only)\s+(\d+)\s+(?:articles|results)', prompt_text_lower):
        summary_result_count = min(int(match.group(1)), 20)
    elif 'essay' in prompt_text_lower or 'all results' in prompt_text_lower:
        summary_result_count = 20
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