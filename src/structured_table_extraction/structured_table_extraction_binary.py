from typing import List, Dict
import json
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

from src.structured_table_extraction.similarity_metrics import embedding_similarity_mpnet_base, jaccard_similarity_wordwise, regular_expression_complete_word_matching
from src.structured_table_extraction import create_excel_accepting_matrix
from src.classes.Query_Constraints import QueryConstraint

class StructuredTableExtraction(): 

    def __init__(self, query_constraints: List[QueryConstraint], filter_max_elements: callable = (lambda x, kwags: x)):
        """_summary_

        Args:
            queries (List[Dict]): List of dict with "query", "similarity_metric", and "dimension".
        """
        self.query_constraints = query_constraints
        if filter_max_elements is None:
            self.filter_max_elements = lambda x, kwags: x
        else: 
            self.filter_max_elements = filter_max_elements
    
    def _df_to_matrix(self, table: pd.DataFrame) -> np.ndarray:
        
        table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
        table_matrix[table_matrix == "None"] = ''
        table_matrix[table_matrix == None] = ''
        table_matrix = table_matrix.astype(str)

        return table_matrix

    def apply_query(self, table_matrix: np.ndarray, query: str, similarity_metric: str, dimension: str):
        """_summary_

        Args:
            table_matrix (np.ndarray): _description_
            accepting_matrix (np.ndarray): _description_
            query (str): _description_
            similarity_metric (str): _description_
            dimension (str): _description_
        """

        temp_accepting_matrix = np.zeros(table_matrix.shape)

        # apply the similarity metric with the query to each element in the table
        
        # create lambda function to calculate similarity for given query
        calculate_similarity = lambda cell: similarity_metric(str(query), str(cell))

        # create similarity matrix
        #similarity_matrix = np.vectorize(calculate_similarity)(table_matrix)
        similarity_matrix = np.array([calculate_similarity(cell) for cell in table_matrix.flatten()]).reshape(table_matrix.shape)

        # adapt the accepting matrix

        # get the indices of the maximum similarity if there is one > 0 
        if np.max(similarity_matrix) > 0:
            max_indicies = np.argwhere(similarity_matrix == np.max(similarity_matrix))
            maximum = np.max(similarity_matrix)
        else:
            return temp_accepting_matrix, similarity_matrix

        # add the maximum similarity to the tempory accepting matrix

        if dimension == "row":

            indices_of_dimension = np.unique(max_indicies[:, 0])
            temp_accepting_matrix[indices_of_dimension] = temp_accepting_matrix[indices_of_dimension] + maximum

        elif dimension == "column":

            indices_of_dimension = np.unique(max_indicies[:, 1])
            temp_accepting_matrix[:, indices_of_dimension] = temp_accepting_matrix[:, indices_of_dimension] + maximum

        elif dimension == "cell":

            temp_accepting_matrix[max_indicies[:, 0], max_indicies[:, 1]] = temp_accepting_matrix[max_indicies[:, 0], max_indicies[:, 1]] + maximum

        return temp_accepting_matrix, similarity_matrix


    def extract(self, table: pd.DataFrame) -> List[Dict]:

        # init log
        log = {"similarity_matrices": [], 
               "max_elements": [], 
               "max_indices": [],
               "accepting_matrix": None, 
               "emission_matrix": None}

        # create table matrix and accepting matrix

        table_matrix = self._df_to_matrix(table)
        accepting_matrix = np.zeros(table_matrix.shape)

        # iterate over queries and apply them to the table matrix

        for query_constraint in self.query_constraints:
            
            temp_query = query_constraint.query
            temp_similarity_metric = query_constraint.similarity_metric
            temp_dimension = query_constraint.dimension

            temp_accepting_matrix, similarity_matrix = self.apply_query(
                table_matrix=table_matrix, 
                query=temp_query, 
                similarity_metric=temp_similarity_metric, 
                dimension=temp_dimension
            )

            accepting_matrix = accepting_matrix + temp_accepting_matrix
        
            log["similarity_matrices"].append({"query": temp_query, "similarity_metric": temp_similarity_metric, "similarity_matrix": similarity_matrix})

        # create log

        # Use accepting matrix to determine results
        
        if np.max(accepting_matrix) == 0:
            log["accepting_matrix"] = accepting_matrix
            log["emission_matrix"] = table_matrix
            return None, log

        max_value = np.max(accepting_matrix)
        max_indices = np.argwhere(accepting_matrix == max_value)

        # get the elements of the table with the max value in the accepting matrix
        max_elements = table_matrix[max_indices[:, 0], max_indices[:, 1]]
        # filter max elements and max indices
        max_elements_filtered = self.filter_max_elements(max_elements, self.query_constraints)

        indices_filtered = [np.where(max_elements == element)[0][0] for element in max_elements_filtered]
        max_indices_filtered = max_indices[indices_filtered]

        # build log and return 
        if len(max_elements_filtered) == 1: 
            result = max_elements_filtered[0]
        else: 
            result = None
        
        log["max_elements_before_filter"] = max_elements
        log["max_elements"] = max_elements_filtered
        log["max_indices_before_filter"] = max_indices
        log["max_indices"] = max_indices_filtered
        log["accepting_matrix"] = accepting_matrix
        log["emission_matrix"] = table_matrix

        return result, log


if __name__ == "__main__": 
    path = "/Users/hendrikweichel/projects/ReMeDi/remedi/Data/test_data/Data_prior_knowledge/Dataset_4/companies_list_4_x_y_with_tables_edenai_temp.csv"
    df = pd.read_csv(path, sep=";", index_col=0)

    i = 7

    table = pd.DataFrame(json.loads(df.iloc[i]["x"]))
    # print(table.to_markdown())
    year = df.iloc[i]["t"]
    emission_type = df.iloc[i]["emission_type"]

    # print(emission_type)

    structuredTableExtraction = StructuredTableExtraction(queries=[
        {"query": year, "similarity_metric": embedding_similarity_mpnet_base.calculate_similarity, "dimension": 1}, 
        {"query": emission_type, "similarity_metric": jaccard_similarity_wordwise.calculate_similarity, "dimension": 0}
        ])

    result, log = structuredTableExtraction.extract(table)

    # store log in execl 

    create_excel_accepting_matrix.create_colored_excel(log["emission_matrix"], log["accepting_matrix"], similarity_matrices=log["similarity_matrices"], params=df.iloc[i].to_dict())
