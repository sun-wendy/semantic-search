from data.animals import ALL_CATEG, ALL_CATEG_NAMES


def get_word_categories(word: str):
    # Check if a word is in any category in ALL_CATEG, return a list of categories
    word_categories = []
    for category in ALL_CATEG:
        if word in category:
            word_categories.append(ALL_CATEG_NAMES[ALL_CATEG.index(category)])
    return word_categories
