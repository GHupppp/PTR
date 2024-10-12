import json
import openai
import os
import time
import re
import argparse
from typing import List, Dict
from openai import OpenAI
# ----------------------- Configuration -----------------------

# Replace 'your-api-key' with your actual OpenAI API key.
# It's recommended to use environment variables for security.
#openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure you set this in your environment

# File paths for json1 and json2
JSON1_PATH = 'ToolLens_tool.json'  # Tool definitions
JSON2_PATH = 'ToolLens_QT.json'  # Queries and tools
OUTPUT_PATH = 'ToolLens_evaluation_results.json'  # Optional: To save detailed results

# OpenAI API settings
MODEL = 'gpt-4o'  # You can change to 'gpt-3.5-turbo' if needed
MAX_RETRIES = 5  # Number of retries for API calls in case of failures
SLEEP_TIME = 2  # Seconds to wait before retrying after a failure

# ----------------------- Few-Shot Examples -----------------------

FEW_SHOT_EXAMPLES = [
    {
        "query": "I need the latest weather forecast for New York and a reminder to carry an umbrella.",
        "tools": ["WeatherTool", "ReminderTool"],
        "classification": "Exact Solving"
    },
    {
        "query": "Show me the top-rated restaurants nearby and provide a route to get there.",
        "tools": ["RestaurantFinder", "RoutePlanner"],
        "classification": "Exact Solving"
    },
    {
        "query": "Find me a good book to read and suggest a nearby coffee shop.",
        "tools": ["BookRecommender", "WeatherTool"],
        "classification": "Partial Solving"
    },
    {
        "query": "Provide the current exchange rates and set a reminder to check them later.",
        "tools": ["FinanceTool", "ReminderTool", "NewsTool"],
        "classification": "Oversolving"
    },
    {
        "query": "I want to track my fitness goals and get news updates.",
        "tools": ["FitnessTracker", "NewsTool"],
        "classification": "Exact Solving"
    },
    {
        "query": "Schedule a meeting and find the latest sports news.",
        "tools": ["CalendarTool", "NewsTool", "FinanceTool"],
        "classification": "Oversolving"
    }
]

# ----------------------- Helper Functions -----------------------

def load_json(file_path: str):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path: str):
    """Save JSON data to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def construct_tools_description(tools: List[str], tool_definitions: Dict[str, str]) -> str:
    """Construct a string of tool descriptions."""
    descriptions = []
    for tool in tools:
        desc = tool_definitions.get(tool, "No description available.")
        descriptions.append(f"- **{tool}**: {desc}")
    return "\n".join(descriptions)

def parse_classification(response: str) -> str:
    """
    Parses the model's response to extract the classification.
    If the response contains one of the expected classes, return it.
    Otherwise, return "Unclear Classification".
    """
    # Define the valid classes
    valid_classes = ["Exact Solving", "Oversolving", "Partial Solving"]
    
    # Use regular expressions to search for the classes
    for cls in valid_classes:
        pattern = re.compile(re.escape(cls), re.IGNORECASE)
        if pattern.search(response):
            return cls
    
    return "Unclear Classification"

def call_openai_api(prompt: str) -> str:
    """Call OpenAI API with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            os.environ["OPENAI_API_KEY"] = ""
            openai.api_key = os.environ["OPENAI_API_KEY"]

            client = OpenAI()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that categorizes tool effectiveness based on given queries and tool sets."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,  # For deterministic output
                max_tokens=60,  # Limit to ensure concise response
                n=1,
                stop=None
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(f"Rate limit reached. Sleeping for {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
        except openai.error.APIError as e:
            print(f"API error: {e}. Retrying in {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"Unexpected error: {e}. Retrying in {SLEEP_TIME} seconds...")
            time.sleep(SLEEP_TIME)
    return "Error: Failed to get a response from OpenAI API."

def evaluate_solving(query: str, tools: List[str], tool_definitions: Dict[str, str], few_shot_text: str) -> str:
    """
    Evaluate if the tools can exactly solve, oversolve, or partially solve the query.
    Returns one of: "Exact Solving", "Oversolving", "Partial Solving", or "Unclear Classification"
    """
    tools_description = construct_tools_description(tools, tool_definitions)
    prompt = f"""
{few_shot_text}

**Query:** "{query}"

**Tools:**
{tools_description}

**Classification:** (Respond with only one of the following exact phrases: "Exact Solving", "Oversolving", or "Partial Solving". Do not include any additional text or explanations.)
"""
    response = call_openai_api(prompt)
    classification = parse_classification(response)
    return classification

def evaluate_solving_after_deletion(query: str, tools: List[str], tool_definitions: Dict[str, str], few_shot_text: str) -> str:
    """
    Evaluate if removing one tool leads to partial solving.
    Returns one of: "Exact Solving", "Oversolving", "Partial Solving", or "Unclear Classification"
    """
    if not tools:
        return "Partial Solving"

    # Remove one tool (e.g., the last tool to minimize impact)
    modified_tools = tools[:-1] if len(tools) > 1 else []

    tools_description = construct_tools_description(modified_tools, tool_definitions)
    prompt = f"""
{few_shot_text}

**Query:** "{query}"

**Tools after removing one tool:**
{tools_description}

**Classification:** (Respond with only one of the following exact phrases: "Exact Solving", "Oversolving", or "Partial Solving". Do not include any additional text or explanations.)
"""
    response = call_openai_api(prompt)
    classification = parse_classification(response)
    return classification

def evaluate_queries(json1: str, json2: str, percentage: float = 10.0) -> Dict:
    """
    Evaluate a subset of queries in json2 based on the tools from json1.
    The subset is determined by the specified percentage.
    Returns a dictionary with detailed results and the final percentage.
    """
    global tool_definitions
    tool_definitions = load_json(json1)
    queries = load_json(json2)
    
    total_queries = len(queries)
    subset_size = max(1, int(total_queries * (percentage / 100)))
    
    # Slice the first 'subset_size' queries
    subset_queries = queries[:subset_size]
    
    print(f"Total Queries in json2: {total_queries}")
    print(f"Processing the first {percentage}% ({subset_size} queries) of json2.\n")
    
    # Construct few-shot examples text
    few_shot_text = "Few-Shot Examples:\n\n"
    for example in FEW_SHOT_EXAMPLES:
        tools_description = "\n".join([f"- **{tool}**: {tool_definitions.get(tool, 'No description available.')}" for tool in example["tools"]])
        few_shot_text += f"**Query:** \"{example['query']}\"\n\n**Tools:**\n{tools_description}\n\n**Classification:** {example['classification']}\n\n"
    
    results = []
    score_1_count = 0
    processed = len(subset_queries)
    
    for idx, item in enumerate(subset_queries, 1):
        query = item.get("query", "")
        tools = item.get("tool", [])
        
        print(f"Evaluating Query {idx}/{processed}: {query}")
        print(f"Tools Used: {tools}")
        
        # First Evaluation
        first_eval = evaluate_solving(query, tools, tool_definitions, few_shot_text)
        print(f"First Evaluation: {first_eval}")
        
        # Second Evaluation
        second_eval = evaluate_solving_after_deletion(query, tools, tool_definitions, few_shot_text)
        print(f"Second Evaluation (After Deletion): {second_eval}")
        
        # Scoring
        if first_eval == "Exact Solving" and second_eval == "Partial Solving":
            score = 1
            score_1_count += 1
        else:
            score = 0
            # Optionally, handle "Unclear Classification"
            if first_eval == "Unclear Classification" or second_eval == "Unclear Classification":
                print("Warning: Unclear classification encountered. Assigning score 0.")
        
        print(f"Score for this query: {score}\n")
        
        results.append({
            "query": query,
            "tools_used": tools,
            "first_evaluation": first_eval,
            "second_evaluation_after_deletion": second_eval,
            "score": score
        })
        
        # To comply with rate limits, consider adding a short delay
        time.sleep(1)  # Adjust as needed based on your rate limits
    
    percentage_score_1 = (score_1_count / processed) * 100 if processed > 0 else 0
    print(f"Processed Queries: {processed}")
    print(f"Queries with Score 1: {score_1_count}")
    print(f"Percentage of Score 1: {percentage_score_1:.2f}%")
    
    results_summary = {
        "total_queries_processed": processed,
        "score_1_count": score_1_count,
        "percentage_score_1": percentage_score_1,
        "detailed_results": results
    }
    
    # Optional: Save detailed results to a JSON file
    save_json(results_summary, OUTPUT_PATH)
    print(f"Detailed results saved to {OUTPUT_PATH}")
    
    return results_summary

# ----------------------- Argument Parsing -----------------------

def parse_arguments():
    """
    Parse command-line arguments to allow dynamic selection of the percentage of queries to process.
    """
    parser = argparse.ArgumentParser(description="Evaluate tool effectiveness on a subset of queries.")
    parser.add_argument(
        '--percentage',
        type=float,
        default=10.0,
        help='Percentage of queries in json2 to evaluate (default: 10.0)'
    )
    parser.add_argument(
        '--json1',
        type=str,
        default=JSON1_PATH,
        help='Path to json1 file containing tool definitions (default: json1.json)'
    )
    parser.add_argument(
        '--json2',
        type=str,
        default=JSON2_PATH,
        help='Path to json2 file containing queries and tools (default: json2.json)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_PATH,
        help='Path to save the evaluation results (default: evaluation_results.json)'
    )
    return parser.parse_args()

# ----------------------- Execution -----------------------

if __name__ == "__main__":
    args = parse_arguments()
    
    # Validate the percentage input
    if not (0 < args.percentage <= 100):
        print("Error: Percentage must be between 0 and 100.")
        exit(1)
    
    # Update paths based on arguments
    JSON1_PATH = args.json1
    JSON2_PATH = args.json2
    OUTPUT_PATH = args.output
    
    # Ensure that the JSON files exist
    if not os.path.exists(JSON1_PATH):
        print(f"Error: {JSON1_PATH} does not exist.")
        exit(1)
    if not os.path.exists(JSON2_PATH):
        print(f"Error: {JSON2_PATH} does not exist.")
        exit(1)
    
    # Run the evaluation with the specified percentage
    evaluation_results = evaluate_queries(JSON1_PATH, JSON2_PATH, percentage=args.percentage)
    
