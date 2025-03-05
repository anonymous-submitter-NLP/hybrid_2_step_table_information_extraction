from typing import List

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

    # If there is a specification of the type (e.g. "Location" in "Scope 1 Location"), separate this
    emission_type_to_list = lambda x: [x] if x in ["Scope 1", "Scope 2", "Scope 3"] else ["Scope 2", x.split(" ")[-1]] if "Scope 2" in x else [x]
    queries = emission_type_to_list(query)

    return float(all([query.lower() in cell.lower() for query in queries]))


# print(calculate_similarity(["Scope 1", "Location"], "Scope 1 Bla bla Market"))