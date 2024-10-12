import numpy as np

def dcg_at_k(relevance_scores, k):
    """
    Calculate Discounted Cumulative Gain (DCG) at rank k.
    Args:
        relevance_scores (list): List of relevance scores for the items.
        k (int): Rank at which DCG is calculated.
    Returns:
        float: DCG value at rank k.
    """
    relevance_scores = np.asarray(relevance_scores)[:k]
    return np.sum(relevance_scores / np.log2(np.arange(2, relevance_scores.size + 2)))


def ndcg_at_k(relevance_scores, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at rank k.
    Args:
        relevance_scores (list): List of relevance scores for the items.
        k (int): Rank at which NDCG is calculated.
    Returns:
        float: NDCG value at rank k.
    """
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    dcg = dcg_at_k(relevance_scores, k)
    idcg = dcg_at_k(ideal_relevance_scores, k)
    return dcg / idcg if idcg > 0 else 0


def calculate_ndcg_for_query(predicted, ground_truth):
    """
    Calculate NDCG for a single query using the size of the predicted list as k.
    Args:
        predicted (list of str): Predicted results for the query.
        ground_truth (list of str): Ground truth results for the query.
    Returns:
        float: NDCG value for this query.
    """
    k = len(ground_truth)  # Use the size of the predicted list as k
    relevance_scores = [1 if item in ground_truth else 0 for item in predicted]
    return ndcg_at_k(relevance_scores, k)


def calculate_average_ndcg(predicted_lists, ground_truth_lists):
    """
    Calculate the average NDCG across multiple queries using predicted list size as k.
    Args:
        predicted_lists (list of list of str): Predicted results for all queries.
        ground_truth_lists (list of list of str): Ground truth results for all queries.
    Returns:
        float: Average NDCG value across all queries.
    """
    ndcg_scores = []
    for predicted, ground_truth in zip(predicted_lists, ground_truth_lists):
        ndcg = calculate_ndcg_for_query(predicted, ground_truth)
        ndcg_scores.append(ndcg)

    return np.mean(ndcg_scores)

def average_length_difference(list1, list2):
    """
    Calculate the average difference in the number of elements between each pair of corresponding lists.

    Args:
        list1 (list of list of str): The first list of lists.
        list2 (list of list of str): The second list of lists.

    Returns:
        float: The average difference in the number of elements between corresponding lists.
    """
    if len(list1) != len(list2):
        raise ValueError("The two input lists must have the same number of sublists.")

    total_difference = 0
    for sublist1, sublist2 in zip(list1, list2):
        total_difference += abs(len(sublist1) - len(sublist2))

    average_difference = total_difference / len(list1)
    return average_difference

def TRACCAndAcc(A, B):
    # Convert both lists to sets for intersection operation
    set_A = set(A)
    set_B = set(B)

    # Calculate the intersection of the two sets
    intersection_AB = set_A.intersection(set_B)

    # Calculate n1 (number of elements in A) and n2 (number of elements in B)
    n1 = len(A)
    n2 = len(B)

    # Calculate the original accuracy
    original = len(intersection_AB) / n1

    # Calculate the union of the two sets
    union_AB = set_A.union(set_B)

    # Calculate the final modified accuracy based on your formula
    TRACC = (1 - (1 / len(union_AB)) * abs(n2 - n1)) * original

    return TRACC, original



"""
# Example usage:
A = ["apple", "banana", "orange", "sss"]
B = ["apple", "orange", "grape"]

TRACC, acc = TRACCAndAcc(A, B)
print(TRACC)
print(acc)
"""
