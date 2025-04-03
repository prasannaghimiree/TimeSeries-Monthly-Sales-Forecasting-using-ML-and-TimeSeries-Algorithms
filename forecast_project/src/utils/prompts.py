def get_query_parser_prompt(query, current_month, current_year, next_month, next_year):
    return f"""
    Analyze the following user query and determine if it is asking about sales or upcoming sales.
    If it is, convert it into a structured JSON format that includes:
    - "type": "single" for a single month, "range" for a range of months, or "quarter" for a quarter
    - "month": the month name (e.g., "Baisakh") for single queries
    - "start_month" and "end_month": for range queries
    - "quarter": the quarter number (1, 2, 3, or 4) for quarter queries
    - "year": the year if specified, otherwise null
    - "operation": "total", "sum", "average", or "mean" if a calculation is requested, otherwise "none"
    - "original": the original query string
    If the query is not about sales or upcoming sales, return:
    - "type": "invalid"
    - "original": the original query string

    Current Nepali month: {current_month} ({current_year})
    Next Nepali month: {next_month} ({next_year})
    Valid months: Baisakh, Jestha, Ashad, Shrawan, Bhadra, Ashoj, Kartik, Mangsir, Poush, Magh, Falgun, Chaitra
    Quarters:
    - Quarter 1: Shrawan (04), Bhadra (05), Ashoj (06)
    - Quarter 2: Kartik (07), Mangsir (08), Poush (09)
    - Quarter 3: Magh (10), Falgun (11), Chaitra (12)
    - Quarter 4: Baisakh (01), Jestha (02), Ashad (03)
    Notes:
    - {current_month} is considered future because the model does not include it as input.
    - If the query contains sales-related keywords (e.g., "sales in Shrawan", "forecasted values", "total sales"), process it.
    - If the query asks for "all months" or "every month," treat it as a range from the first to the last forecast month.
    - For "next 2 months" or similar, include {current_month} and {next_month}.
    - Detect mathematical operations like "total", "sum", "average", or "mean" and include them in the "operation" field.

    Query: "{query}"

    Examples:
    - "forecast me the sales of baisakh" → {{"type": "single", "month": "Baisakh", "year": null, "operation": "none", "original": "forecast me the sales of baisakh"}}
    - "total sales from baisakh to shrawan" → {{"type": "range", "start_month": "Baisakh", "end_month": "Shrawan", "year": null, "operation": "total", "original": "total sales from baisakh to shrawan"}}
    - "average sales in quarter 1" → {{"type": "quarter", "quarter": "1", "year": null, "operation": "average", "original": "average sales in quarter 1"}}
    - "2082 quarter 4 sales" → {{"type": "quarter", "quarter": "4", "year": "2082", "operation": "none", "original": "2082 quarter 4 sales"}}
    - "what's the weather like" → {{"type": "invalid", "original": "what's the weather like"}}
    - "total sales for next 2 months" → {{"type": "range", "start_month": "{current_month}", "end_month": "{next_month}", "year": null, "operation": "total", "original": "total sales for next 2 months"}}
    """