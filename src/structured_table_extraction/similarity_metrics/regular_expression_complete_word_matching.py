

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
    #print(float(query.lower() in cell.lower()))
    #print(query, cell, float(query.lower() in cell.lower()))

    return float(query.lower() in cell.lower())