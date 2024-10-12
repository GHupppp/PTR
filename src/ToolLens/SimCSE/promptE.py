# Import necessary libraries
import openai
import os
# Set your OpenAI API key
from openai import OpenAI
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
def call_mistral_api(prompt: str) -> str:
    os.environ["MISTRAL_API_KEY"] = ""
    api_key = os.environ["MISTRAL_API_KEY"]

    client = MistralClient(api_key=api_key)
    response = client.chat(
        model="open-mistral-7b",  # or 'gpt-3.5-turbo' if you don't have access to GPT-4
        messages=[
            ChatMessage(role= "user", content =prompt)
        ],
        max_tokens=1024,
        temperature=0  # Use 0 for deterministic output
    )

    return response.choices[0].message.content.strip()



def call_openai_api(prompt: str) -> str:
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    os.environ["OPENAI_API_KEY"] = ""
    openai.api_key = os.environ["OPENAI_API_KEY"]

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",  # or 'gpt-3.5-turbo' if you don't have access to GPT-4
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0  # Use 0 for deterministic output
    )

    return response.choices[0].message.content.strip()

def extract_key_requirements(query: str) -> list:
    """
    Extracts key requirements from the original query using prompt engineering with Few-Shot Learning.
    """
    prompt = f"""
You are an AI assistant helping to extract key requirements from user queries.

Example 1:
User Query:
"I want a website where users can create accounts, post messages, and follow other users."

Key Requirements:
- Users can create accounts
- Users can post messages
- Users can follow other users

Example 2:
User Query:
"I need an e-commerce platform that supports product listings, shopping cart functionality, payment processing, and order tracking."

Key Requirements:
- Supports product listings
- Provides shopping cart functionality
- Handles payment processing
- Offers order tracking

Now, given the following user query, extract the key requirements.

User Query:
"{query}"

Key Requirements:
"""
    response = call_openai_api(prompt)
    # Process the response into a list
    key_requirements = [line.strip('- ').strip() for line in response.split('\n') if line.strip()]
    return key_requirements

def match_tools_to_requirements(requirements: list, tools: dict) -> dict:
    print(tools)
    """
    Matches tools to key requirements using the language model with Few-Shot Learning.
    """
    tool_matches = {}
    for req in requirements:
        prompt = f"""
You are an AI assistant helping to match tools to requirements, as long as the tool description can roughly provid the needed information for requirments, it does not need to be very specific,ignore the proper nouns.

Available Tools:
"""
        for tool_name, tool_desc in tools.items():
            prompt += f"- **{tool_name}**: {tool_desc}\n"

        prompt += f"""
Example 1:
Requirement:
"I want to know the latest news about Tesla"

Matched Tools:
- NewsTool: Stay connected to global events with our up-to-date news around the world.

Example 2:
Requirement:
"Please provide me with the current stock price of Apple"

Matched Tools:
- FinanceTool: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more.

Now, for the following requirement, list the tools from the available tools that can fulfill it.

Requirement:
"{req}"

Matched Tools:
"""
        response = call_openai_api(prompt)
        # Process the response to extract matched tools
        matched_tools = []
        for line in response.split('\n'):
            line = line.strip('- ').strip()
            if line.startswith('Tool'):
                tool_name = line.split(':')[0].strip()
                matched_tools.append(tool_name)
        tool_matches[req] = matched_tools
    return tool_matches

def assess_toolset_completeness(tool_matches: dict) -> tuple:
    """
    Assesses the toolset completeness and identifies unsolved problems.
    """
    tools_to_keep = set()
    unsolved_requirements = []
    for req, matched_tools in tool_matches.items():
        if matched_tools:
            tools_to_keep.update(matched_tools)
        else:
            unsolved_requirements.append(req)
    is_exact_solve = len(unsolved_requirements) == 0
    return is_exact_solve, list(tools_to_keep), unsolved_requirements

def identify_unsolved_problems(unsolved_requirements: list) -> list:
    """
    Identifies unsolved problems directly from the original query.
    """
    unsolved_problems = unsolved_requirements.copy()
    return unsolved_problems

def main(original_query, tools):
    # Original query
    """
    original_query = (
        "I need to develop an online educational platform where instructors can create courses, "
        "students can enroll, watch video lectures, submit assignments, participate in discussions, "
        "and receive certificates upon completion."
    )

    # Initial toolset with descriptions
    tools = {
        "Tool A": "A Learning Management System (LMS) framework that supports course creation, "
                  "student enrollment, user authentication, certificate issuance, and basic grading functionalities.",
        "Tool B": "A video streaming service that securely hosts video content and provides APIs for integration.",
        "Tool C": "A discussion forum module that facilitates threaded discussions among users.",
        "Tool V": "An e-commerce plugin that handles payment processing for course enrollment fees."
    }
    """
    # Step 1: Extract Key Requirements
    key_requirements = extract_key_requirements(original_query)
    print("Key Requirements Extracted:")
    for req in key_requirements:
        print(f"- {req}")
    print()

    # Step 2: Match Tools to Key Requirements
    tool_matches = match_tools_to_requirements(key_requirements, tools)
    print("Tool Matches:")
    for req, matched_tools in tool_matches.items():
        tools_list = ', '.join(matched_tools) if matched_tools else "None"
        print(f"- Requirement: '{req}' matched with Tools: {tools_list}")
    print()

    # Step 3: Assess Toolset Completeness
    is_exact_solve, tools_to_keep, unsolved_requirements = assess_toolset_completeness(tool_matches)
    print(f"Does the toolset exactly solve the query? {'Yes' if is_exact_solve else 'No'}")
    print(f"Tools to Keep: {', '.join(tools_to_keep)}")
    print()

    # Step 4: Identify Unsolved Problems
    unsolved_problems = identify_unsolved_problems(unsolved_requirements)
    print("Unsolved Problems:")
    for problem in unsolved_problems:
        print(f"- {problem}")
    print()

    return tools_to_keep, unsolved_problems
    # The unsolved_problems list can now be used in your rerank method

if __name__ == "__main__":
    main()

