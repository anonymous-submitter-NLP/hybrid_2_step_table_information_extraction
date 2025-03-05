from src.formatting import formatting


def calculate_similarity(query: str, cell: str) -> float: 
    """Create a function to calculate the similarity between two strings. 

    Here, we 
    
    1. generate for both the query and the cell a number that is contained in the string.
    2. Calculate the difference
    3. Return 0 if the difference is larger than 1, else 1
    
    Return a float value between 0 and 1.

    Args:
        query (str): Query, that is compared to the cell
        cell (str): Cell value
        n (int, optional): Length of string element. Defaults to 2.

    Returns:
        float: _description_
    """

    # convert the cell to float 
    cell_quantity = formatting.transform_emissions_to_float(cell)["quantity"]

    # convert the query to float
    query_quantity = formatting.transform_emissions_to_float(query)["quantity"]

    if cell_quantity is None or query_quantity is None: 
        return 0
    
    if query_quantity != 0:

        result = max(1 - abs(abs(cell_quantity - query_quantity) / query_quantity), 0)

    else: 

        result = max(1 - abs(cell_quantity - query_quantity), 0)

    # if result != 0 and result is not None:
    #     print(f"Cell: {cell_quantity}")
    #     print(f"Query: {query_quantity}")
    #     print(f"Result: {result}")
    #     print()
    #     print()
    #     print()
    #     print()

    return result