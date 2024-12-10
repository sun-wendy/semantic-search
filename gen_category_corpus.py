import re

from utils import get_word_categories, get_all_animal_words


animal_words = get_all_animal_words()


def generate_category_corpus(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()

    # Match blocks using the numbered pattern
    pattern = r"(\d+-----\nSENTENCE: .+?\nSEM_REP: .+?)(?=\n\d+-----|\Z)"
    blocks = re.findall(pattern, data, flags=re.S)
    updated_blocks = []

    for block in blocks:
        # Extract SENTENCE & SEM_REP
        sentence_match = re.search(r"SENTENCE: (.+)", block)
        sem_rep_match = re.search(r"SEM_REP: (.+)", block)
        
        if sentence_match and sem_rep_match:
            sentence = sentence_match.group(1)
            sem_rep = sem_rep_match.group(1)
            
            # Split the sentence into words
            words_in_sentence = re.findall(r"\b\w+\b", sentence.lower())
            
            # Find the animal word in the sentence
            animal = next((word for word in words_in_sentence if word in animal_words), None)
            if animal:
                # Get categories for the animal
                categories = get_word_categories(animal)
                # Replace SEM_REP with categories
                new_sem_rep = "," + ",".join(categories)
                # Update the block
                updated_block = re.sub(r"SEM_REP: .+", f"SEM_REP: {new_sem_rep}", block)
                updated_blocks.append(updated_block)

    # Write updated content to the output file
    with open(output_file, 'w') as file:
        file.write("\n\n".join(updated_blocks))


if __name__ == "__main__":
    generate_category_corpus("results/training_corpus/llm_corpus.txt", "results/training_corpus/category_corpus.txt")
