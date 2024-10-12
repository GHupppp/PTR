import openai
import json
import itertools
import random
import os
from tqdm import tqdm
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
# ----------------------------- Configuration -----------------------------

# Set your OpenAI API key as an environment variable for security
# Alternatively, you can directly assign it as a string (not recommended)
#openai.api_key = os.getenv("OPENAI_API_KEY")

# If you prefer to set the API key directly (less secure), uncomment the line below and replace 'your-api-key'
# openai.api_key = 'your-api-key'

# Configuration Parameters
TOOL_JSON = {
    "FinanceTool": "Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more.",
    "ExchangeTool": "Seamlessly convert currencies with our integrated currency conversion tool.",
    "NewsTool": "Stay connected to global events with our up-to-date news around the world.",
    "PolishTool": "Elevate your content with our AI-powered tool, which utilizes advanced rewriting techniques to create more human-like expressions and foster creative inspiration.",
    "CharityTool": "Empower your charitable endeavors by accessing a comprehensive platform featuring nonprofit organization data, including their mission, key personnel, ratings, and financial information.",
    "MapTool": "Experience the next level of map navigation with our innovative chatbot, leveraging Google Maps API to generate customized map images based on location, tilt, and style, and even annotate maps using latitude and longitude coordinates.",
    "CourseTool": "Unlock a world of knowledge and growth with our comprehensive learning platform, offering a diverse range of courses from renowned providers like Coursera and Upskillr, personalized language learning, professional team information lookup, open course schedule discovery, and top-tier university content.",
    "DataRetrievalTool": "An tool that expands memory. It stores and retrieves user information to provide personalized assistance and generates real-time chat responses using the knowledge from the document collection.",
    "StrologyTool": "Provides astrology services for you.",
    "NotesTool": "A full-featured reminder and to-do list management tool where you can add, delete, list, and mark reminders.",
    "MemoryTool": "A learning application with spaced repetition functionality that allows users to create flashcards and review them.",
    "LawTool": "Enables quick search functionality for relevant laws.",
    "ChartTool": "A versatile chart and diagram tool that can create and display diagrams or using networkx and matplotlib. It allows you to build, modify, and draw various charts and graphs within the chat interface.",
    "EarthquakeTool": "Provides real-time earthquake notifications and news.",
    "NASATool": "A platform for exploring space, allowing users to search and discover NASA images and utilize NASA's vast media library.",
    "CompanyInfoTool": "Obtain relevant information about global companies from databases or knowledge graphs.",
    "ResumeTool": "Quickly create resumes and receive feedback on your resume.",
    "MemeTool": "Create memes.",
    "GiftTool": "Provide suggestions for gift selection.",
    "PDF&URLTool": "Interact with any PDF files, provide page references for fact-checking, support chatting via Google Drive links to AI-driven PDF summaries and analysis; engage in interactive conversations with websites, access links on the internet to fetch required information, including generating articles and intelligent assistance for interactions with source code.",
    "VideoSummarizeTool": "Generate summaries from YouTube video links, offer question-answering capabilities, analyze and interpret the content of YouTube videos, and support interactions with online video platforms such as YouTube and Daily Motion.",
    "MediaModifyTool": "A versatile image editing application with a vast selection of user-generated filters, allowing you to effortlessly edit photos and videos. It includes embedded features such as resizing, cropping, and blurring.",
    "DietTool": "A tool that simplifies calorie counting, tracks diet, and provides insights from many restaurants and grocery stores. Explore recipes, menus, and cooking tips from millions of users, and access recipe consultations and ingredient delivery services from thousands of stores.",
    "WebsiteTool": "Quickly create and deploy websites, and publish content on them.",
    "URLTool": "Provide domain and URL information and assist users with domain registration.",
    "TripTool": "Offer discounted hotel and accommodation bookings, along with personalized hotel and product searches, travel planning, image editing, and more, helping users easily plan their trips and find accommodation and transportation options.",
    "TripAdviceTool": "A comprehensive travel assistant that makes travel planning more vivid and practical. It offers tourism activities, accommodation and attraction recommendations, aiming to provide users with a more enjoyable and enriching travel experience through technology.",
    "BookTool": "AI-powered personalized book recommendations, access to free children's picture books, and the ability to search and create books on Wikidocs.",
    "MediaTool": "Your media and entertainment companion, offering recommendations for TV shows, movies, books, and podcasts, while providing access to free ad-supported content across the web.",
    "PodcastTool": "Search for podcasts and summarize their content.",
    "MusicTool": "Create music playlists, search for music, and check out the latest music trends.",
    "GameTool": "Get game-related information and recommend games.",
    "WeatherTool": "Provide you with the latest weather information.",
    "RestaurantBookingTool": "Tool for booking restaurant",
    "local": "Discover and support restaurants, shops & services near you.",
    "HouseRentingTool": "Tool that provides all sorts of information about house renting",
    "HousePurchasingTool": "Tool that provides all sorts of information about house purchasing",
    "JobTool": "Your Global Career Hub! Find diverse job opportunities, expert interview tips, and resume optimization guidance. Empowering job seekers worldwide on their path to success.",
    "RepoTool": "Discover GitHub projects tailored to your needs, explore their structures with insightful summaries, and get quick coding solutions with curated snippets. Elevate your coding journey with RepoTool, your go-to companion for GitHub project exploration and code mastery.",
    "ResearchFinder": "Tool for searching academic papers.",
    "ResearchHelper": "Tool that offers additional functions beyond searching academic papers, such as generating mind maps, answering user questions and storing them in specific formats.",
    "SEOTool": "Tool that provides users with SEO analytics content.",
    "ProductSearch": "Find products tailored to your preferences with personalized recommendations and smart filters for specific needs.",
    "Discount": "Discover discounts and coupon codes to save money on products.",
    "Review": "Analyze and summarize reviews, providing advantages and disadvantages analysis of products.",
    "ProductComparison": "Compare multiple product options for informed decisions.",
    "ShoppingAssistant": "Manage your cart, and display QR code for easy cart to enhance your online shopping experience."
}

# Updated list of combination sizes
COMBINATION_SIZES = [1,2,3,4,5,6,7,8,9,10]

# Number of unique combinations per size
COMBINATIONS_PER_SIZE = 20

# Number of queries per combination
QUERIES_PER_COMBINATION = 5

# Output JSON file path
OUTPUT_FILE = "query_tools_dataset.json"

# OpenAI GPT-4 model to use
GPT_MODEL = "gpt-4o"

# Maximum number of retries for API calls
MAX_RETRIES = 5

# Delay between retries (in seconds)
RETRY_DELAY = 5

# Delay between successful API calls to respect rate limits
SUCCESS_DELAY = 1

# --------------------------------------------------------------------------


def generate_tool_combinations(tools, combination_size, num_combinations):
    """
    Generates a list of unique tool combinations.

    Args:
        tools (list): List of tool names.
        combination_size (int): Number of tools in each combination.
        num_combinations (int): Number of unique combinations to generate.

    Returns:
        list of tuples: Each tuple is a unique combination of tools.
    """
    all_combinations = list(itertools.combinations(tools, combination_size))
    if len(all_combinations) < num_combinations:
        raise ValueError(f"Not enough unique combinations for size {combination_size}. Requested: {num_combinations}, Available: {len(all_combinations)}")
    selected_combinations = random.sample(all_combinations, num_combinations)
    return selected_combinations


def prepare_few_shot_examples():
    """
    Prepares a list of few-shot examples to guide GPT-4.

    Returns:
        list of dict: Each dict contains a 'query' and 'tool' list.
    """
    # These examples cover combination sizes from 1 to 10
    few_shot_examples = [
        # Single Tool Example
        {
            "query": "I want to know the latest news about Tesla.",
            "tool": ["NewsTool"]
        },
        # 2-Tool Examples
        {
            "query": "Convert 500 GBP to USD and analyze the current exchange rates trends.",
            "tool": ["ExchangeTool", "FinanceTool"]
        },
        {
            "query": "Track your international expenses by converting currencies and managing your budget.",
            "tool": ["ExchangeTool", "FinanceTool"]
        },
        # 3-Tool Examples
        {
            "query": "Plan a trip to Italy, book hotels, and find the best local restaurants.",
            "tool": ["TripTool", "HouseRentingTool", "local"]
        },
        {
            "query": "Organize a corporate retreat, secure accommodations, and find nearby dining options.",
            "tool": ["TripTool", "HouseRentingTool", "local"]
        },
        # 4-Tool Examples
        {
            "query": "Search for academic papers on climate change, generate a mind map, and summarize the findings.",
            "tool": ["ResearchFinder", "ResearchHelper", "VideoSummarizeTool", "NotesTool"]
        },
        {
            "query": "Find research articles on renewable energy, create a visual mind map, and extract key insights.",
            "tool": ["ResearchFinder", "ResearchHelper", "VideoSummarizeTool", "NotesTool"]
        },
        # 5-Tool Examples
        {
            "query": "Track my daily tasks, set reminders for important deadlines, and analyze my productivity trends.",
            "tool": ["NotesTool", "RemindersTool", "ChartTool", "FinanceTool", "MemoryTool"]
        },
        {
            "query": "Manage my to-do list, receive deadline notifications, and visualize my work patterns.",
            "tool": ["NotesTool", "RemindersTool", "ChartTool", "FinanceTool", "MemoryTool"]
        },
        # 6-Tool Examples
        {
            "query": "Plan a comprehensive diet plan, set meal reminders, track your calorie intake, and analyze nutritional data.",
            "tool": ["DietTool", "NotesTool", "RemindersTool", "ChartTool", "FinanceTool", "Review"]
        },
        {
            "query": "Design a balanced meal schedule, receive meal alerts, monitor your daily calories, and assess your dietary progress.",
            "tool": ["DietTool", "NotesTool", "RemindersTool", "ChartTool", "FinanceTool", "Review"]
        },
        # 7-Tool Examples
        {
            "query": "Plan your trip, book accommodations, find local restaurants, track your budget, set travel reminders, manage your itinerary, and get weather updates.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool"]
        },
        {
            "query": "Organize a business trip, secure lodging, discover dining spots, monitor expenses, schedule travel alerts, outline your itinerary, and check the weather forecast.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool"]
        },
        # 8-Tool Examples
        {
            "query": "Convert currencies, analyze financial data, fetch the latest news, visualize trends on a map, manage expenses, set financial reminders, track budget allocations, and generate financial reports.",
            "tool": ["ExchangeTool", "FinanceTool", "NewsTool", "MapTool", "NotesTool", "RemindersTool", "ChartTool", "DataRetrievalTool"]
        },
        {
            "query": "Manage international transactions, evaluate financial performance, stay updated with global events, map spending patterns, oversee expenditures, receive budget alerts, allocate funds effectively, and compile financial summaries.",
            "tool": ["ExchangeTool", "FinanceTool", "NewsTool", "MapTool", "NotesTool", "RemindersTool", "ChartTool", "DataRetrievalTool"]
        },
        # 9-Tool Examples
        {
            "query": "Plan an extensive international trip, book flights and hotels, find top local restaurants, manage your budget, set travel reminders, track your itinerary, get weather updates, visualize travel routes on a map, and generate travel summaries.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool", "MapTool", "VideoSummarizeTool"]
        },
        {
            "query": "Organize a global vacation, reserve flights and accommodations, discover local eateries, monitor your spending, set travel alerts, outline your daily schedule, check weather conditions, map your travel routes, and compile a travel journal.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool", "MapTool", "VideoSummarizeTool"]
        },
        # 10-Tool Examples
        {
            "query": "Plan a global vacation, book flights and hotels, find top local restaurants, manage your budget, set travel reminders, track your itinerary, get weather updates, visualize travel routes on a map, generate travel summaries, and create a travel blog.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool", "MapTool", "VideoSummarizeTool", "MediaTool"]
        },
        {
            "query": "Organize an international getaway, reserve transportation and lodging, explore local dining options, oversee your finances, schedule travel alerts, outline your daily activities, monitor weather forecasts, map your journey, summarize your experiences, and document your travels online.",
            "tool": ["TripTool", "HouseRentingTool", "local", "FinanceTool", "RemindersTool", "NotesTool", "WeatherTool", "MapTool", "VideoSummarizeTool", "MediaTool"]
        }
    ]
    return few_shot_examples


def construct_prompt(few_shot_examples, tool_names, tool_descriptions):
    """
    Constructs the prompt to send to GPT-4 for query generation using few-shot learning.

    Args:
        few_shot_examples (list of dict): List of example queries and their tool combinations.
        tool_names (list): List of tool names in the current combination.
        tool_descriptions (dict): Dictionary mapping tool names to their descriptions.

    Returns:
        str: The constructed prompt with few-shot examples.
    """
    # Construct few-shot examples
    examples_text = ""
    for example in few_shot_examples:
        examples_text += f"""Tool Combination: {example['tool']}
Query: {example['query']}

"""

    # Describe the task with additional instructions for uniqueness
    task_description = f"""You are an AI assistant tasked with generating user queries that can be exclusively solved by a specific set of tools.

**Requirements for the query:**
1. The query must **only** require the functionalities of the selected tools.
2. All tools in the selected set must be **necessary** to solve the query.
3. The query should **not** require any tools outside the selected set.
4. The query should be **clear, specific, and realistic**.
5. **Each query should address a different scenario or aspect** to ensure uniqueness. Avoid merely rephrasing similar ideas; focus on varied use cases.

**Selected Tools:**
{', '.join(tool_names)}

**Tool Descriptions:**
"""
    for name in tool_names:
        task_description += f"- **{name}**: {tool_descriptions[name]}\n"

    task_description += """

Generate one unique query that meets the above requirements.

**Query:**"""

    # Combine examples and task
    prompt = examples_text + task_description

    return prompt


def is_unique(new_query, existing_queries, threshold=0.8):
    """
    Checks if the new_query is unique compared to existing_queries based on cosine similarity.

    Args:
        new_query (str): The newly generated query.
        existing_queries (list of str): List of existing queries for the tool combination.
        threshold (float): Similarity threshold to consider queries as duplicates.

    Returns:
        bool: True if unique, False otherwise.
    """
    if not existing_queries:
        return True
    queries_embeddings = model.encode(existing_queries)
    new_query_embedding = model.encode([new_query])
    similarities = cosine_similarity(new_query_embedding, queries_embeddings)[0]
    return max(similarities) < threshold


def call_gpt4_with_uniqueness(prompt, existing_queries, model_name=GPT_MODEL, max_tokens=150, temperature=0.8, presence_penalty=0.5, frequency_penalty=0.5):
    """
    Calls the GPT-4 API and ensures the generated query is unique.

    Args:
        prompt (str): The prompt to send to the model.
        existing_queries (list of str): Existing queries for the tool combination.
        model_name (str): The model to use.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): Sampling temperature.
        presence_penalty (float): Presence penalty.
        frequency_penalty (float): Frequency penalty.

    Returns:
        str: A unique generated query.

    Raises:
        Exception: If a unique query cannot be generated within the maximum retries.
    """
    for attempt in range(MAX_RETRIES):
        try:
            os.environ["OPENAI_API_KEY"] = ""
            openai.api_key = os.environ["OPENAI_API_KEY"]

            client = OpenAI()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are ChatGPT, a large language model trained by OpenAI."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            generated_query = response.choices[0].message.content.strip()
            
            # Clean the query
            if "Query:" in generated_query:
                generated_query = generated_query.split("Query:")[-1].strip()
            elif "query:" in generated_query:
                generated_query = generated_query.split("query:")[-1].strip()
            
            # Check for uniqueness
            if is_unique(generated_query, existing_queries):
                return generated_query
            else:
                print("Duplicate query detected. Regenerating...")
                time.sleep(RETRY_DELAY)
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error: {e}. Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
    raise Exception("Max retries exceeded for generating a unique OpenAI API call.")


# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def main():
    # Extract tool names and descriptions
    tool_names_all = list(TOOL_JSON.keys())
    tool_descriptions = TOOL_JSON

    # Prepare few-shot examples
    few_shot_examples = prepare_few_shot_examples()

    dataset = []
    # Dictionary to keep track of existing queries per tool combination
    queries_per_combination = {}

    for size in COMBINATION_SIZES:
        print(f"\nGenerating {COMBINATIONS_PER_SIZE} unique tool combinations of size {size}...")
        try:
            combinations = generate_tool_combinations(tool_names_all, size, COMBINATIONS_PER_SIZE)
        except ValueError as ve:
            print(f"Error: {ve}")
            continue

        for combo in tqdm(combinations, desc=f"Processing {size}-tool combinations"):
            combo_key = tuple(sorted(combo))  # Sort to ensure consistency
            queries_per_combination[combo_key] = []

            for i in range(QUERIES_PER_COMBINATION):
                
                prompt = construct_prompt(few_shot_examples, combo, tool_descriptions)
                print(prompt)
                try:
                    query = call_gpt4_with_uniqueness(prompt, queries_per_combination[combo_key])
                    print(query)
                    dataset.append({
                        "query": query,
                        "tool": list(combo)
                    })
                    queries_per_combination[combo_key].append(query)
                    # Optional: Add a short delay to respect rate limits
                    time.sleep(SUCCESS_DELAY)
                except Exception as e:
                    print(f"Failed to generate a unique query for tools {combo}: {e}")

    # Optionally, include few-shot examples in the dataset
    # Uncomment the following lines if you want to include them
    """
    for example in few_shot_examples:
        dataset.insert(0, example)
    """

    # Save the dataset to a JSON file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset generation complete. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

