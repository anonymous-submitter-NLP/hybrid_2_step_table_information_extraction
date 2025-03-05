from typing import List, Dict
import json
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

from src.structured_table_extraction.similarity_metrics import embedding_similarity_mpnet_base, jaccard_similarity_wordwise, numerical_comparison_continuous_no_match, regular_expression_complete_word_matching
from src.structured_table_extraction import create_excel_accepting_matrix
from src.structured_table_extraction import structured_table_extraction_helper
from src.classes.Query_Constraints import QueryConstraint

class StructuredTableExtractionProb(): 

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
        
        self.log = {}
        self.log["priority_levels"] = {query_constraint.priority for query_constraint in query_constraints}

    def create_similarity_matrix(self, query_constraint: QueryConstraint, table_matrix: np.ndarray):
        """_summary_

        Args:
            query_constraint (QueryConstraint): _description_
            table_matrix (np.ndarray): _description_
        """
        
        # define input parameters
        temp_query = query_constraint.query
        temp_similarity_metric = query_constraint.similarity_metric
        calculate_similarity = lambda cell: temp_similarity_metric(str(temp_query), str(cell))

        # create similarities for each cell in the table
        similarity_matrix = np.array([calculate_similarity(cell) for cell in table_matrix.flatten()]).reshape(table_matrix.shape)

        return similarity_matrix

    def create_similarity_matrices(self, table: pd.DataFrame) -> pd.DataFrame: 

        self.similarity_matrices = []

        table_matrix = structured_table_extraction_helper.df_to_matrix(table)
        self.table_matrix = table_matrix
        self.log["emission_matrix"] = table_matrix

        for query_constraint in self.query_constraints:
            
            temp_similarity_matrix = {}
            temp_similarity_matrix["priority"] = query_constraint.priority
            temp_similarity_matrix["dimension"] = query_constraint.dimension
            temp_similarity_matrix["similarity_metric"] = query_constraint.similarity_metric
            temp_similarity_matrix["query"] = query_constraint.query
            temp_similarity_matrix["similarity_matrix"] = self.create_similarity_matrix(query_constraint, table_matrix)

            self.similarity_matrices.append(temp_similarity_matrix)
            
        self.log["similarity_matrices"] = (self.similarity_matrices)

        self.similarity_matrices = pd.DataFrame(self.similarity_matrices)
    
        return self.similarity_matrices

    def evaluata_similarity_query_constrains_only(self): 
        """ 
        
        Based on the similarity metrics and queries provided, return a ranking for all cells with scorings for all priority classes.
        
        """

        # record the accepting matrix of every priority level
        accepting_matrices = []
    
        # record the scores of eqch priority level

        # get a list of all cell indices
        cell_indices = np.indices(self.table_matrix.shape).reshape(2, -1).T
        df_scores = pd.DataFrame({"Cell_content": self.table_matrix.flatten(), "Cell_index_row": cell_indices[:,0], "Cell_index_column": cell_indices[:,1]})

        # loop over all priority classes
        self.similarity_matrices = self.similarity_matrices.sort_values(by=["priority"])
        for priority in self.similarity_matrices["priority"].unique():

            temp_similarity_matrices = self.similarity_matrices[self.similarity_matrices["priority"] == priority]

            # aggregate matrices

            aggregation_matrix = np.zeros(self.table_matrix.shape)

            for i, similarity_matrix_row in temp_similarity_matrices.iterrows():
                
                temp_dimension = similarity_matrix_row["dimension"]

                similarity_matrix_row_only_max = np.where(similarity_matrix_row["similarity_matrix"] == similarity_matrix_row["similarity_matrix"].max(), similarity_matrix_row["similarity_matrix"], 0)

                if temp_dimension == "row":
                    max_elements_row = np.array(similarity_matrix_row["similarity_matrix"]).max(axis=1)
                    temp_similarity_matrix = np.tile(max_elements_row[:, np.newaxis], similarity_matrix_row["similarity_matrix"].shape[1]) - similarity_matrix_row_only_max
                elif temp_dimension == "column":
                    max_elements_column = np.array(similarity_matrix_row["similarity_matrix"]).max(axis=0)
                    temp_similarity_matrix = np.tile(max_elements_column, (similarity_matrix_row["similarity_matrix"].shape[0], 1)) - similarity_matrix_row_only_max
                elif temp_dimension == "cell":
                    temp_similarity_matrix = similarity_matrix_row["similarity_matrix"]

                aggregation_matrix = aggregation_matrix + temp_similarity_matrix
            
            accepting_matrices.append({"priority": priority, "aggregation_matrix": aggregation_matrix})

            df_scores[f"Scores_priority_{priority}"] = aggregation_matrix.flatten()
        
        df_scores = df_scores.sort_values(by=[f"Scores_priority_{priority}" for priority in self.similarity_matrices["priority"].unique()], ascending=False).reset_index(drop=True)
        max_elements_df = df_scores.copy()

        # obtain highest results
        max_elements_per_priority  = []
        for priority in self.similarity_matrices["priority"].unique():
            max_elements_df = max_elements_df[max_elements_df[f"Scores_priority_{priority}"] == max_elements_df[f"Scores_priority_{priority}"].max()]
            max_elements_per_priority.append({"priority": priority, "max_elements": max_elements_df})
        
        # fill logs
        
        max_elements = max_elements_per_priority[-1]["max_elements"]

        self.log['max_elements'] = max_elements["Cell_content"].tolist()
        self.log['max_indices'] = np.array([max_elements["Cell_index_row"].tolist(), max_elements["Cell_index_column"].tolist()]).T
        
        # fill other level max elements
        for max_elements_prio in max_elements_per_priority: 
            priority = max_elements_prio["priority"]
            max_elements_prio = max_elements_prio["max_elements"]
            self.log[f'max_elements_prio_{priority}'] = max_elements_prio["Cell_content"].tolist()
            self.log[f'max_indices_prio_{priority}'] = np.array([max_elements_prio["Cell_index_row"].tolist(), max_elements_prio["Cell_index_column"].tolist()]).reshape((len(max_elements_prio), 2))
            self.log[f'max_scores_prio_{priority}'] = max_elements_prio
        
        self.log["accepting_matrices"] = accepting_matrices
        self.log["df_scores"] = df_scores

        return max_elements["Cell_content"].tolist(), df_scores

    def evaluata_similarity_matrices_llm(self): 
        pass
    
    def evaluata_similarity_matrices_tapas(self): 
        pass

    def extract(self, table: pd.DataFrame, selection_method: str = "query_constrains_only") -> pd.DataFrame:
        """
        
        Selection Methods: 
            - "query_constrains_only"
            - "llm"
            - "tapas"

        """ 

        # 1. Create similarity matrices

        self.create_similarity_matrices(table)

        # 2. Evaluate similarity matrices

        if selection_method == "query_constrains_only":
            max_element, df_scores = self.evaluata_similarity_query_constrains_only()
            return max_element, self.log 
            
        elif selection_method == "llm":
            max_element, df_scores = self.evaluata_similarity_matrices_llm()
            return max_element, df_scores
        
        elif selection_method == "tapas":
            max_element, df_scores = self.evaluata_similarity_matrices_tapas()
            return max_element, df_scores


if __name__ == "__main__": 
    path = "/Users/hendrikweichel/projects/ReMeDi/remedi/Data/test_data/Data_prior_knowledge/Dataset_4/companies_list_4_x_y_with_tables_edenai_temp.csv"
    df = pd.read_csv(path, sep=";", index_col=0)

    i = 10

    table = pd.DataFrame(json.loads(df.iloc[i]["x"]))
    # print(table.to_markdown())
    year = df.iloc[i]["t"]
    emission_type = df.iloc[i]["emission_type"]
    prior_y = df.iloc[i]["prior_y"]

    print(prior_y)

    # structuredTableExtraction = StructuredTableExtraction(queries=[
    #     {"query": year, "similarity_metric": embedding_similarity.calculate_similarity, "dimension": 1}, 
    #     {"query": emission_type, "similarity_metric": embedding_similarity.calculate_similarity, "dimension": 0}
    #     ])
    
    query_constraints = [
        QueryConstraint(year, regular_expression_complete_word_matching.calculate_similarity, "column", 1), 
        QueryConstraint(emission_type, regular_expression_complete_word_matching.calculate_similarity, "row", 1),
        QueryConstraint(prior_y, numerical_comparison_continuous_no_match.calculate_similarity, "cell", 2)
    ]

    structuredTableExtraction = StructuredTableExtractionProb(query_constraints=query_constraints)
    
    max_element, log = structuredTableExtraction.extract(table)

    accepting_matrix = log["accepting_matrices"][0]["aggregation_matrix"]

    #print(log["similarity_matrices"])

    # 'similarity_matrices', 'max_elements', 'max_indices', 'accepting_matrix', 'emission_matrix', 'max_elements_before_filter', 'max_indices_before_filter'

    create_excel_accepting_matrix.create_colored_excel(log["emission_matrix"], accepting_matrix, similarity_matrices=log["similarity_matrices"], params=df.iloc[i].to_dict(), df_scores=log["df_scores"])
