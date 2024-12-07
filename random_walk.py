import numpy as np
from typing import List, Tuple, Set
import os
import csv

from incremental_network import SemanticNetworkLearner


class RandomWalkSearch:
    def __init__(self, network_learner: SemanticNetworkLearner):
        self.network = network_learner.network
        self.word_meanings = network_learner.word_meanings
        self.clusters = network_learner.clusters
    

    def random_walk(self, start_word: str = "animal", num_steps: int = 70) -> List[Tuple[str, Set[str]]]:
        """
        Perform random walk through the network starting from given word.
        Returns list of (word, categories) tuples.
        """
        if start_word not in self.network:
            raise ValueError(f"Start word '{start_word}' not in network")
        
        walk_seq = []
        cur_word = start_word
        
        for _ in range(num_steps):
            # Get neighbors and their edge weights
            neighbors = list(self.network.neighbors(cur_word))
            if not neighbors:
                print(f"Warning: No neighbors for word '{cur_word}', returning to start word")
                cur_word = start_word
                continue
            
            # Get edge weights as transition probabilities
            weights = [self.network[cur_word][neighbor]['weight'] for neighbor in neighbors]
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Choose next word based on edge weights
            next_word = np.random.choice(neighbors, p=weights)
            
            # Record word and its categories
            walk_seq.append((next_word, self.word_meanings[next_word]))
            cur_word = next_word
            
        return walk_seq


def save_walk_results(walk_seq: List[Tuple[str, Set[str]]], filename: str = 'outputs/random_walk/random_walk_5.csv'):
    """Save walk results to a CSV file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Word", "Categories"])
        for i, (word, categories) in enumerate(walk_seq, 1):
            writer.writerow([i, word, ", ".join(sorted(categories))])


if __name__ == "__main__":
    # Initialize & train network
    learner = SemanticNetworkLearner(rho=0.8, rho_animal=0.4)
    learner.process_corpus_file('outputs/learning_corpus.txt')
    learner.squash_edge_weights(alpha=0.7)
    
    # Perform random walk
    walker = RandomWalkSearch(learner)
    walk_seq = walker.random_walk()
    
    # Save results
    save_walk_results(walk_seq)
    
    # Print first 10 steps
    print("\nFirst 10 steps of random walk:")
    for i, (word, categories) in enumerate(walk_seq[:10], 1):
        print(f"{i}. {word}: {', '.join(sorted(categories))}")
    
    # Print summary statistics
    unique_words = len(set(word for word, _ in walk_seq))
    print(f"\nTotal steps: {len(walk_seq)}")
    print(f"Unique words visited: {unique_words}")
