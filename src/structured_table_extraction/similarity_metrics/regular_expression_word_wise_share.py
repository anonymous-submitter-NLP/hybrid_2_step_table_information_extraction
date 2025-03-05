
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

    query_words = query.split(" ")

    return sum([query_word.lower() in cell.lower() for query_word in query_words]) / len(query_words)

if __name__ == "__main__":

    # Example usage
    query_string = "Scope 2 Market"

    cells = ["Total Scope 2 (t CO2e) electricity purchased - Location Based",
    "Total Scope 2 (t CO2e) thermal energy purchased - Location Based",
    "Total Scope 2 (t CO2e) - Location Based",
    "Total Scope 2 (t CO2e) electricity purchased - Market Based",
    "Total Scope 2 (t CO2e) thermal energy purchased - Market Based",
    "Total Scope 2 (t CO2e) - Market Based"]

    similarity_score = [calculate_similarity(query_string, cell_string) for cell_string in cells]
    for i in range(len(cells)):
        print(f"Similarity between '{query_string}' and '{cells[i]}': {similarity_score[i]}")