import random
from typing import List, Tuple, Dict, Set

# Import animal categories from animals.py
from animals import (african_animals, animals_used_for_fur, arctic_far_north_animals,
                    australian_animals, beasts_of_burden, birds, bovine, canine,
                    deers, farm_animals, feline, fish, insectivores, insects,
                    north_american_animals, pets, primates, rabbits,
                    reptiles_amphibians, rodents, water_animals, weasels,
                    ALL_CATEG_NAMES, ALL_CATEG)

def load_base_vocabulary() -> Set[str]:
    """Load base vocabulary from animals.txt"""
    with open('animals.txt', 'r') as f:
        return {word.strip().lower() for word in f.readlines()}

def get_animal_features(animal: str) -> Set[str]:
    """Get semantic features (categories) for an animal"""
    features = set()
    for category, animals in zip(ALL_CATEG_NAMES, ALL_CATEG):
        if animal.lower() in [a.lower() for a in animals]:
            features.add(category)
            # Add parent categories
            if category in ['african_animals', 'arctic_far_north_animals', 
                          'australian_animals', 'north_american_animals']:
                features.add('REGION')
            if category in ['canine', 'feline', 'bovine']:
                features.add('FAMILY')
            if category in ['pets', 'farm_animals', 'beasts_of_burden']:
                features.add('DOMESTIC')
            features.add('animal')  # Add root category
    return features

def generate_utterance() -> List[str]:
    """Generate a simple utterance containing an animal word"""
    templates = [
        ["look", "at", "the", "{animal}"],
        ["i", "see", "a", "{animal}"],
        ["there", "is", "a", "{animal}"],
        ["the", "{animal}", "is", "here"],
        ["a", "{animal}", "walks", "by"]
    ]
    return random.choice(templates)

def generate_corpus(num_examples: int, 
                   output_file: str = "training_corpus.txt") -> None:
    """Generate a corpus of utterance-scene pairs"""
    base_vocab = load_base_vocabulary()
    
    with open(output_file, 'w') as f:
        for i in range(num_examples):
            # Select random animal from base vocabulary
            animal = random.choice(list(base_vocab))
            
            # Generate utterance
            utterance_template = generate_utterance()
            utterance = [word.replace("{animal}", animal) 
                        for word in utterance_template]
            
            # Get scene features
            features = get_animal_features(animal)
            
            # Write to file in format:
            # 1-----
            # SENTENCE: look at the dog
            # SEM_REP: ,animal,canine,pets,DOMESTIC,FAMILY
            f.write(f"{i+1}-----\n")
            f.write(f"SENTENCE: {' '.join(utterance)}\n")
            f.write(f"SEM_REP: {','.join([''] + list(features))}\n\n")

if __name__ == "__main__":
    # Generate 1000 training examples
    generate_corpus(1000)
    print("Corpus generation complete!")