

def calculate_similarity(query: str, cell: str) -> float: 
    """Create a function to calculate the similarity between two strings. 
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    get_numerical_chars = lambda s: ''.join(filter(str.isdigit, s))

    return float(get_numerical_chars(query) in get_numerical_chars(cell))

#print(calculate_similarity("2,300", "23100"))