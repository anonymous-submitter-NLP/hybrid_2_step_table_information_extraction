from typing import List

def filter_not_numerical_values(x: List[str], kwags) -> List[str]:
    """Takes a list of strings and filters out the itmes of them contain no numerical values

    Args:
        x (List[str]): _description_

    Returns:
        List[str]: _description_
    """

    return [i for i in x if any(c.isdigit() for c in i)]