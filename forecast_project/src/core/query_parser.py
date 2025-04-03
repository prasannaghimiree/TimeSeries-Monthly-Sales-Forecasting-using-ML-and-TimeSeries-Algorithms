import json
import logging
import nepali_datetime
from src.utils.constants import MONTH_NUMBER_TO_NAME
from src.utils.prompts import get_query_parser_prompt

logger = logging.getLogger(__name__)

class QueryParser:
    def __init__(self, llm_model):
        self.model = llm_model

    def parse(self, query):
        # function to parse prompt. It links with prompts.py
        current_date = nepali_datetime.date.today()
        cur_year, cur_month_num = current_date.year, current_date.month
        current_month = MONTH_NUMBER_TO_NAME[f"{cur_month_num:02d}"]
        # if month is 12(chaitra), next month is 1(Baisakh) else just add +1 to it
        next_month_num = 1 if cur_month_num == 12 else cur_month_num + 1
        next_year = cur_year + 1 if cur_month_num == 12 else cur_year
        next_month = MONTH_NUMBER_TO_NAME[f"{next_month_num:02d}"]

        prompt = get_query_parser_prompt(query, current_month, cur_year, next_month, next_year)
        try:
            response = self.model.generate_content(prompt)
            return json.loads(response.text.strip("```json\n").strip("```"))
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {"type": "invalid", "original": query}