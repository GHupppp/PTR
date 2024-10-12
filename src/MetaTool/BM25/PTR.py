import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import Counter
import openai
import os
from openai import OpenAI
import re
import promptE
import metric
from rank_bm25 import BM25Okapi
import random
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from sentence_transformers import util
#return the tool lists in the form"Tool: tool description"
def getToolCombinations(file_name, tool_descriptions):
    with open(file_name, 'r') as file:
        data = json.load(file)

    # Create a set to store unique tool combinations
    tool_combinations = set()

    # Iterate through each item in the JSON data
    for item in data:
        # Sort the tool list to ensure order does not matter
        sorted_tools = tuple(sorted(item['tool']))
        # Add the sorted tuple to the set
        tool_combinations.add(sorted_tools)

    # Convert the set back to a list of lists
    unique_combinations = [list(combination) for combination in tool_combinations]

    # Print or return the unique combinations
    sentences = []
    for combination in tool_combinations:
        sentence = "+".join([f"{tool}: {tool_descriptions[tool]}" for tool in combination])
        sentences.append(sentence)
    return sentences

def testAndtrain(groundTruth_file_name):
    with open(groundTruth_file_name, 'r') as f:
        groundTruth = json.load(f)
    queries_test = []
    tools_test = []

    # Iterate through each item in the JSON data
    test_size = int(len(groundTruth) / 10)
    for item in groundTruth[:100]: #only test 100 examples --------------------------------------**************************
        queries_test.append(item['query'])
        tools_test.append(item['tool'])

    queries_train = []
    tools_train = []
    query_tool_dict_train = {}

    # Process each item in the JSON data
    for item in groundTruth[100:]:
        # Extract the content after "query" and remove the surrounding quotes
        query = item["query"].strip(':')
        queries_train.append(query)

        # Extract the content after "tool" and add the list to tools
        query_tool_list = item["tool"]
        tools_train.append(query_tool_list)

        # Populate the dictionary
        query_tool_dict_train[query] = query_tool_list
    return queries_test, tools_test, queries_train, tools_train, query_tool_dict_train

#return two items, the first is a tool list, the format is "tool:description", the second is a tool dictionary
def toolDescription(toolDescription_file_name):
    with open(toolDescription_file_name, 'r') as f:
        data_tool = json.load(f)
    # Initialize lists to hold queries and tools

    tool_list = []
    tool_dict = {}

    # Iterate through each item in the JSON data
    for key, value in data_tool.items():
        # Combine the key and value into a single string
        combined_string = f"{key}: {value}"
        tool_list.append(combined_string)
        tool_dict[key] = value
    return tool_list, tool_dict

def unique_tool(queries, query_tool_dict_train):# Initialize a set to store unique items
    unique_tools = set()

    # Iterate over the queries and add corresponding tools to the set
    for query in queries:
        unique_tools.update(query_tool_dict_train[query])

    # Convert the set back to a list (if required)
    unique_tools_list = list(unique_tools)
    return unique_tools_list


def find_toolCandidate(queriesTrain, tool_list, queriesTest, query_tool_dict_train):
    HuggingFace_embedding = HuggingFaceEmbeddings()  # 向量长度--768
    embedding_model = HuggingFace_embedding
    vectordb_QueryTrain = FAISS.from_texts(texts=queriesTrain, embedding=embedding_model)
    vectordb_Tool = FAISS.from_texts(texts=tool_list, embedding=embedding_model)

    potential_tool_list = []
    for query in queriesTest:
        simi_search_tool = vectordb_Tool.similarity_search(query, 7)
        tool_list1 = [docu.page_content for docu in simi_search_tool]
        top_tool = tool_list1[0]
        extracted_toolName1 = [item.split(":")[0] for item in tool_list1]
#        print(extracted_toolName1)

        simi_search_Query = vectordb_QueryTrain.similarity_search(query, 7)
        Query_list = [docu.page_content for docu in simi_search_Query]
        unique = unique_tool(Query_list, query_tool_dict_train)
#        print(unique)

        simi_search_ToolTool = vectordb_Tool.similarity_search(top_tool, 7)
        toolTool_list = [docu.page_content for docu in simi_search_ToolTool]
        extracted_toolName2 = [item.split(":")[0] for item in toolTool_list]
#       print(extracted_toolName2)

        combined_list = extracted_toolName1 + unique + extracted_toolName2

# Count the frequency of each element
        element_counts = Counter(combined_list)

# Sort elements by their frequency (from high to low)
        sorted_elements = [item for item, count in element_counts.most_common()]

# Print the result
        potential_tool_list.append(sorted_elements)
    return potential_tool_list


# Different Retriever - BM25, Random, Contriever, Pre_trained model



def Contriever_retrieve_toolBundle(query, toolBundle_list, tool_description):
    # 1. Load the Contriever model from SentenceTransformers
    model = SentenceTransformer('facebook/contriever')


# 3. Encode the query and documents into embeddings
    query_embedding = model.encode([query])[0]  # Single query embedding
    doc_embeddings = model.encode(toolBundle_list)    # Multiple document embeddings

# --- Option 1: Using FAISS for similarity search ---

# 4. Initialize a FAISS index for L2 similarity search
    embedding_size = doc_embeddings.shape[1]  # Embedding size (dimension)
    index = faiss.IndexFlatL2(embedding_size)

# 5. Add document embeddings to the FAISS index
    index.add(np.array(doc_embeddings))

# 6. Perform the search to find the document closest to the query embedding
    D, I = index.search(np.array([query_embedding]), k=1)  # Search for the top 1 result

# 7. Get the most related document index and document
    most_related_doc_idx = I[0][0]
    top_tool = toolBundle_list[most_related_doc_idx]



    entries = top_tool.split('+')

# Extract the tool names
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic




def Random_retrieve_toolBundle(query, toolBundle_list, tool_description):
    top_tool = random.choice(toolBundle_list)
    entries = top_tool.split('+')

# Extract the tool names
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic



def BM25_retrieve_toolBundle(query, toolBundle_list, tool_description):
    tokenized_IO = [doc.split(" ") for doc in toolBundle_list]
    bm25 = BM25Okapi(tokenized_IO)
    tokenized_query = query.split(" ")
    top_tool = bm25.get_top_n(tokenized_query, toolBundle_list, n=1)[0]

    entries = top_tool.split('+')

# Extract the tool names
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic

def SBERT_retrieve_toolBundle(query, toolBundle_list, tool_description):
    HuggingFace_embedding = HuggingFaceEmbeddings()
    embedding_model = HuggingFace_embedding
    vectordb_toolBundle = FAISS.from_texts(texts=toolBundle_list, embedding=embedding_model)
    simi_search_bundle = vectordb_toolBundle.similarity_search(query, 1)
    tool_list3 = [docu.page_content for docu in simi_search_bundle]
    top_tool = tool_list3[0]
    #extracted_toolName3 = [item.split(":")[0] for item in tool_list3]    
    entries = top_tool.split('+')

# Extract the tool names
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic




def SimCSE_retrieve_toolBundle(query, toolBundle_list, tool_description, model_name='princeton-nlp/sup-simcse-bert-base-uncased'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(toolBundle_list, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_result = torch.argmax(cosine_scores).item()
    top_tool = toolBundle_list[top_result]
    similarity_score = cosine_scores[top_result].item()
    entries = top_tool.split('+')
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic


def TAS_B_retrieve_toolBundle(query, toolBundle_list, tool_description, model_name='sentence-transformers/msmarco-distilbert-base-tas-b'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(toolBundle_list, convert_to_tensor=True)
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarities
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]

    # Find the index of the highest similarity score
    top_result = torch.argmax(cosine_scores).item()

    # Retrieve the most similar sentence and its score
    top_tool = toolBundle_list[top_result]
    similarity_score = cosine_scores[top_result].item()

    entries = top_tool.split('+')

# Extract the tool names
    tool_names = [entry.split(':')[0].strip() for entry in entries]
    toolBundle_dic = {}
    for tool_name in tool_names:
        toolBundle_dic[tool_name] = tool_description.get(tool_name)
        #toolBundle_dic[tool_name] = tool_description[tool_name]
    return toolBundle_dic

def main(tool_query_fileName, tool_des_fileName):

    queries_testAll, tools_test, queries_train, tools_train, query_tool_dict_train = testAndtrain(tool_query_fileName)
    tool_list, tool_dict = toolDescription(tool_des_fileName)

    #queries_test = "Please provide me with the current stock price of Apple and any recent news related to the company." ##
    

    toolBundleList = getToolCombinations(tool_query_fileName, tool_dict)
    #bundleDic = Contriever_retrieve_toolBundle(queries_test, toolBundleList, tool_dict) ##
    #print(bundleDic) ##
    #print("done") ##
    #exit() ##
    finalRec_query = []

    for queries_test in tqdm(queries_testAll): 
        bundleDic = BM25_retrieve_toolBundle(queries_test, toolBundleList, tool_dict)
        print(bundleDic)

        toolset1, unsolvedProblem = promptE.main(queries_test, bundleDic)
        print(toolset1)
        print(unsolvedProblem)

        filtered_list = [item for item in toolset1 if not item.lower().startswith('tool')]

        finalToolRec = set(filtered_list)
        toollist1 = find_toolCandidate(queries_train, tool_list, unsolvedProblem, query_tool_dict_train)
        for x in toollist1:
            finalToolRec.add(x[0])
        finalRec_query.append(list(finalToolRec))
    return finalRec_query, tools_test

if __name__ == "__main__":
    final, test = main("MetaTool_QT.json", "MetaTool_tool.json")
    print(final)
    print(test)
    length = len(test)
    sumTRACC = 0
    sumAcc = 0
    for x in range(0, length):
        a,b = metric.TRACCAndAcc(test[x],final[x])
        sumTRACC = sumTRACC + a
        sumAcc = sumAcc + b
    print("NDCG-GroundTruth")
    print(float(metric.calculate_average_ndcg(final, test)))
    print("TRACC")
    print(float(sumTRACC/length))
    print("Recall")
    print(float(sumAcc/length))
    print("Precise")
    print(float(metric.average_length_difference(final,test)))
"""
with open('listsAno.txt', 'w') as file:
    # Write the first list to the file
    file.write(str(tools_test[:10]) + '\n')

    # Write the second list to the file
    file.write(str(toolRec) + '\n')
"""
