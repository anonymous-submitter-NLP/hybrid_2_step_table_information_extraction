import ast
import numpy as np
import pandas as pd
import json


def get_prior_table_row_name(row):

    prior_table_index = np.array(ast.literal_eval(row["prior_y_index"]))

    table = pd.DataFrame(json.loads(row["prior_x"])).astype(str)
    table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
    table_matrix[table_matrix == None] = ''
    table_matrix[table_matrix == "None"] = ''

    for i in range(table_matrix.shape[1]):
        if table_matrix[prior_table_index[0][0], i] != '':
            return table_matrix[prior_table_index[0][0], i]
    
    return table_matrix[prior_table_index[0][0], 0]

def get_prior_table_column_name(row):
    prior_table_index = np.array(ast.literal_eval(row["prior_y_index"]))

    table = pd.DataFrame(json.loads(row["prior_x"])).astype(str)
    table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
    table_matrix[table_matrix == None] = ''
    
    return table_matrix[0, prior_table_index[0][1]]


def get_table_row_name(row):
    prior_table_index = np.array(ast.literal_eval(row["y_index"]))

    table = pd.DataFrame(json.loads(row["x"])).astype(str)
    table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
    table_matrix[table_matrix == None] = ''
    
    return table_matrix[prior_table_index[0][0], 0]

def get_table_column_name(row):
    prior_table_index = np.array(ast.literal_eval(row["y_index"]))

    table = pd.DataFrame(json.loads(row["x"])).astype(str)
    table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
    table_matrix[table_matrix == None] = ''
    
    return table_matrix[0, prior_table_index[0][1]]

def df_to_matrix(table: pd.DataFrame) -> np.ndarray:
      
      table_matrix = np.concatenate([[np.array(table.columns)] ,np.array(table)])
      table_matrix[table_matrix == "None"] = ''
      table_matrix[table_matrix == None] = ''
      table_matrix = table_matrix.astype(str)
      return table_matrix

def rank_df_by_muli_columns(df: pd.DataFrame, sort_by_columns: list) -> pd.DataFrame: 

    df.sort_values(by=sort_by_columns)

    rank = 0
    df.loc[0,"Rank"] = rank
    for i, row in df[1:].iterrows(): 
        rank_up = 0
        for column in sort_by_columns: 
            if row[column] < df.iloc[i - 1][column]: 
                rank_up = 1

        rank = rank + rank_up
        df.loc[i,"Rank"] = rank

    return df


def cosine_similarity_to_0_1(sim: float) -> float: 
    return (sim + 1) / 2