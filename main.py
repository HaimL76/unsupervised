from dimension_reduction import calculate


def main():
    arr_files: list = [
        (r'ds\sleep_deprivation_dataset_detailed.csv',),
        (r'ds\Bank_Transaction_Fraud_Detection.csv',),
        (r'ds\sales_data.csv',),
        (r'ds\sleep_cycle_productivity.csv',),
        (r'ds\car_price_dataset.csv',),
        (r'ds\heart.csv',),
        (r'ds\cardio_train.csv', ['id'], ';'),
        (r'ds\cardio_data_processed.csv', ['id'],),
        (r'ds\alzheimers_disease_data.csv', ['PatientID']),
        (r'ds\health_data.csv', ['id']),
        (r'ds\UserCarData.csv', ['Sales_ID'], ',', (2, 40), 'sold'),
        (r'ds\schizophrenia_dataset.csv', ['Patient_ID'], ',', (2, 10), 'Diagnosis')
    ]

    file_tuple: tuple = arr_files[-1]

    file_path: str = None
    file_separator: str = None
    columns_to_drop: list = None
    k_min = 2
    k_max = 22
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
        target_column = file_tuple[4]

    if file_path:
        if file_separator is None:
            file_separator = ','

        calculate(file_path, target_column=target_column, drop_target_column=False,
                  columns_to_drop=columns_to_drop, csv_sep=file_separator,
                  k_min=k_min, k_max=k_max)


if __name__ == "__main__":
    main()
