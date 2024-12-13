import pandas as pd
import argparse


def calculate_irt(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'step' not in df.columns or 'word' not in df.columns:
        raise ValueError("The CSV file must contain 'step' & 'word' columns.")

    seen_words = set()
    last_seen_index = -1
    irt_values = []

    for index, row in df.iterrows():
        word = row['word']
        step = row['step']

        if word == "animal":
            irt = None
        elif word not in seen_words:
            # Compute IRT for a new unique word
            if last_seen_index != -1:
                irt = step - df.at[last_seen_index, 'step']
            else:
                irt = 1  # For the first word in the sequence
            seen_words.add(word)
            last_seen_index = index
        else:
            irt = None  # IRT is undefined for repeated words in this context

        irt_values.append(irt)

    df['irt'] = irt_values
    df.to_csv(csv_file_path, index=False)
    return df


def add_associative_patch_switch(csv_file_path):
    df = pd.read_csv(csv_file_path)
    if 'categories' not in df.columns or 'word' not in df.columns:
        raise ValueError("The CSV file must contain 'categories' & 'word' columns")

    associative_patch_switch = []
    last_relevant_index = -1

    for index, row in df.iterrows():
        categories = row['categories']
        word = row['word']

        if word == "animal":
            # Skip "animal" and keep the last relevant index unchanged
            associative_patch_switch.append(None)
            continue

        if last_relevant_index == -1:
            # For the first relevant row, no previous row to compare
            associative_patch_switch.append(0)
        else:
            last_categories = df.at[last_relevant_index, 'categories']
            current_categories_set = set(str(categories).split(', '))
            last_categories_set = set(str(last_categories).split(', '))

            # Check if there is no intersection between the sets
            if current_categories_set.isdisjoint(last_categories_set):
                associative_patch_switch.append(1)
            else:
                associative_patch_switch.append(0)

        last_relevant_index = index

    df['associative_patch_switch'] = associative_patch_switch
    df.to_csv(csv_file_path, index=False)
    return df


def add_patch_entry_position(csv_file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    if 'associative_patch_switch' not in df.columns or 'word' not in df.columns:
        raise ValueError("The CSV file must contain 'associative_patch_switch' and 'Word' columns.")

    # Initialize variables
    patch_entry_position = []
    position = None  # Start with None

    # Iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        word = row['word']
        switch = row['associative_patch_switch']

        if word == "animal":
            # Skip "animal" rows entirely
            patch_entry_position.append(None)
            continue

        if switch == 1:
            position = 1  # Reset to 1 for patch switch
        elif position is not None:  # Increment position for consecutive rows
            position += 1

        patch_entry_position.append(position)

    # Add the patch_entry_position column to the DataFrame
    df['patch_entry_position'] = patch_entry_position

    df.to_csv(csv_file_path, index=False)

    return df





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_walk_file', type=str, required=True)
    args = parser.parse_args()

    # updated_df = calculate_irt(args.random_walk_file)
    # updated_df = add_associative_patch_switch(args.random_walk_file)
    updated_df = add_patch_entry_position(args.random_walk_file)
