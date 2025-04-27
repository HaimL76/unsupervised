from dimension_reduction import calculate

list_of_columns: list = [('Age', 0), ('Gender', 1), ('Education_Level', 1), ('Marital_Status', 1),
                         ('Occupation', 1), ('Income_Level', 1), ('Place_of_Residence', 1),
                         ('Substance_Use', 1), ('Social_Support', 1), ('Stress_Factors', 1),
                         ('Family_History_of_Schizophrenia', 1), ('Number_of_Hospitalizations', 0),
                         ('Disease_Duration', 0)]


def main():
    arr_files: list = [
        (r'ds\schizophrenia_dataset.csv', ['Patient_ID'], ',', (2, 60), 'Diagnosis', 'Suicide_Attempt', list_of_columns)
    ]

    file_tuple: tuple = arr_files[-1]

    file_path: str = None
    file_separator: str = None
    columns_to_drop: list = None
    k_min = 2
    k_max = 22
    pivot_column: str = None
    list_columns: list = None
    target_column: str = None

    len_file_tuple = len(file_tuple)

    if len_file_tuple > 0:
        file_path = file_tuple[0]

    if len_file_tuple > 1:
        columns_to_drop = file_tuple[1]

    if len_file_tuple > 2:
        file_separator = file_tuple[2]

    if len_file_tuple > 3:
        k_tup = file_tuple[3]

        if isinstance(k_tup, tuple) and len(k_tup) == 2:
            k_min = k_tup[0]
            k_max = k_tup[1]

    if len_file_tuple > 4:
        pivot_column = file_tuple[4]

    if len_file_tuple > 5:
        target_column = file_tuple[5]

    if len_file_tuple > 6:
        list_columns = file_tuple[6]

    if file_path:
        if file_separator is None:
            file_separator = ','

        calculate(file_path, pivot_column=pivot_column, target_column=target_column, drop_pivot_column=False,
                  columns_to_drop=columns_to_drop, csv_sep=file_separator,
                  k_min=k_min, k_max=k_max, list_of_columns=list_columns)


if __name__ == "__main__":
    main()
