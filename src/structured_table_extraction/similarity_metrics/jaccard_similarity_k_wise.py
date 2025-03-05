
from src.structured_table_extraction.similarity_metrics import regular_expression_complete_word_matching

def calculate_similarity(query: str, cell: str, n=3) -> float:
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    # prefilter with regex

    query = query.lower()
    cell = cell.lower()

    # Helper function to generate n-grams from a string
    def generate_ngrams(s, n):
        # Create a list of n-grams (substrings of length n)
        return {s[i:i+n] for i in range(len(s) - n + 1)}
    
    # Generate n-grams for both strings
    ngrams1 = generate_ngrams(query, n)
    ngrams2 = generate_ngrams(cell, n)
    
    # Calculate intersection and union of the two n-gram sets
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    
    # Compute Jaccard similarity
    jaccard_score = len(intersection) / len(union) if len(union) > 0 else 0
    
    return jaccard_score