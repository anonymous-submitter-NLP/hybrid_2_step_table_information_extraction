"""
Get the Path to the results excel and the folder of all tables and rename to FALSE_RIGHT depending on whether the recall is 0 or 1 

"""
import pandas as pd
import glob
import os

def rename_file(file_path, new_name):
    directory = os.path.dirname(file_path)
    new_path = os.path.join(directory, new_name)
    try: 
        os.rename(file_path, new_path)
    except FileNotFoundError:
        pass
    return new_path

def rename_false_wrong_tables(path_to_results, path_to_tables):

    # get df
    df = pd.read_excel(path_to_results)
    df.head()

    # rename files depending on recall
    df[df["recall"] == 0].apply(lambda row: rename_file(os.path.join(path_to_tables, f'{row["report"]}_{row["emission_type"]}.xlsx'), f'not_found_{row["report"]}_{row["emission_type"]}.xlsx'), axis=1)
    df[(df["recall"] == 1) & (df["len_max_index"] > 1)].apply(lambda row: rename_file(os.path.join(path_to_tables, f'{row["report"]}_{row["emission_type"]}.xlsx'), f'found_multiple_{row["report"]}_{row["emission_type"]}.xlsx'), axis=1)
    df[(df["recall"] == 1) & (df["len_max_index"] == 1)].apply(lambda row: rename_file(os.path.join(path_to_tables, f'{row["report"]}_{row["emission_type"]}.xlsx'), f'found_one_{row["report"]}_{row["emission_type"]}.xlsx'), axis=1)


if __name__ == "__main__": 
    path_to_results = "/Users/hendrikweichel/projects/ReMeDi/remedi/Tests/test_results/test_rule_based_extraction/tables_prior_table_row_name_regular_expression/0_resultsprior_table_row_name_regular_expression.xlsx"
    path_to_tables = "/Users/hendrikweichel/projects/ReMeDi/remedi/Tests/test_results/test_rule_based_extraction/tables_prior_table_row_name_regular_expression/"
