import pandas as pd


def prepare_input_data(raw_input, label_encoders, model_columns):
    input_data = pd.DataFrame([raw_input])

    for col in input_data.columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data[model_columns]
    return input_data