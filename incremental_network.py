import numpy as np
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Set
import re
from pyvis.network import Network
from hdbscan import HDBSCAN
import os
import argparse


class SemanticNetworkLearner:
    def __init__(self, rho: float = 0.8, rho_animal: float = 0.4):
        # Network parameters
        self.rho = rho
        self.rho_animal = rho_animal
        self.word_meanings: Dict[str, Set[str]] = defaultdict(set)
        self.network = nx.Graph()
        
        # Clustering parameters
        self.clusters: Dict[int, Set[str]] = {}
        self.cluster_prototypes: Dict[int, np.ndarray] = {}
        self.word_to_cluster: Dict[str, int] = {}
        self.words_seen: List[str] = []
    

    def process_corpus_file(self, filename: str) -> None:
        """Process the corpus file"""
        cur_sentence, cur_features = [], []

        count = 1
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('SENTENCE:'):
                    cur_sentence = line.replace('SENTENCE:', '').strip().split()
                elif line.startswith('SEM_REP:'):
                    cur_features = [f for f in line.replace('SEM_REP:', '').strip().split(',') if f]
                    if cur_sentence and cur_features:
                        self.process_utterance(cur_sentence, cur_features)
                        print(count)
                        count += 1
    

    def process_utterance(self, utterance: List[str], features: List[str]) -> None:
        """Process a single utterance-scene pair"""
        # Only consider words in animals.txt
        with open('data/animals.txt', 'r') as f:
            animals_list = {word.strip().lower() for word in f.readlines()}
        
        for word in utterance:
            if word in animals_list or word == 'animal':
                if re.match(r'^[a-z]+$', word):  # Simple content word check
                    # Add features to word's meaning
                    self.word_meanings[word].update(features)
                    self._update_network(word)
    

    def _cosine_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words based on their categories"""
        all_categories = set().union(*self.word_meanings.values())
        vec1 = np.array([1 if cat in self.word_meanings[word1] else 0 for cat in all_categories])
        vec2 = np.array([1 if cat in self.word_meanings[word2] else 0 for cat in all_categories])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    

    def _update_clusters(self, word: str) -> None:
        """Update semantic clusters incrementally"""
        self.words_seen.append(word)
        
        if len(self.words_seen) < 10:  # Need minimum number of words for clustering
            return
            
        # Create feature vectors for clustering
        all_categories = sorted(set().union(*self.word_meanings.values()))
        vectors = []
        for w in self.words_seen:
            vec = [1 if cat in self.word_meanings[w] else 0 for cat in all_categories]
            vectors.append(vec)
        
        # Perform clustering
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2)
        cluster_labels = clusterer.fit_predict(vectors)
        
        # Update clusters
        self.clusters.clear()
        self.cluster_prototypes.clear()
        self.word_to_cluster.clear()
        
        for i, label in enumerate(cluster_labels):
            if label >= 0:  # Ignore noise points (-1)
                word = self.words_seen[i]
                if label not in self.clusters:
                    self.clusters[label] = set()
                    self.cluster_prototypes[label] = np.zeros(len(all_categories))
                
                self.clusters[label].add(word)
                self.word_to_cluster[word] = label
                
                # Update prototype (centroid) for this cluster
                self.cluster_prototypes[label] += np.array(vectors[i])
        
        # Normalize prototypes
        for label in self.cluster_prototypes:
            if len(self.clusters[label]) > 0:
                self.cluster_prototypes[label] /= len(self.clusters[label])
    

    def _get_comparison_candidates(self, word: str, n: int = 5) -> Set[str]:
        """Get limited set of words to compare based on clusters"""
        candidates = set()
        
        # Always include neighbors
        candidates.update(self.network.neighbors(word))
        
        # Add words from same cluster
        if word in self.word_to_cluster:
            cluster = self.word_to_cluster[word]
            candidates.update(self.clusters[cluster])
        
        # Add some words from other clusters
        for cluster_id in self.clusters:
            if cluster_id != self.word_to_cluster.get(word):
                cluster_words = list(self.clusters[cluster_id])
                if cluster_words:
                    candidates.add(np.random.choice(cluster_words))
        
        return candidates


    def _update_network(self, updated_word: str) -> None:
        """Update network connections incrementally"""
        if updated_word not in self.network:
            self.network.add_node(updated_word)
            
        # Update clusters
        self._update_clusters(updated_word)
        
        # Get limited set of words to compare
        candidates = self._get_comparison_candidates(updated_word)
        
        # Update existing connections
        for neighbor in list(self.network.neighbors(updated_word)):
            sim = self._cosine_similarity(updated_word, neighbor)
            threshold = self.rho_animal if 'animal' in (updated_word, neighbor) else self.rho
            if sim >= threshold:
                self.network[updated_word][neighbor]['weight'] = sim
            else:
                self.network.remove_edge(updated_word, neighbor)
        
        # Check connections with candidates
        for word in candidates:
            if word != updated_word and not self.network.has_edge(updated_word, word):
                sim = self._cosine_similarity(updated_word, word)
                threshold = self.rho_animal if 'animal' in (updated_word, word) else self.rho
                if sim >= threshold:
                    self.network.add_edge(updated_word, word, weight=sim)
    

    def plot_network(self, filename: str) -> None:
        """Create an interactive HTML visualization of the semantic network with labels"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        net = Network(height='750px', width='100%', bgcolor='#ffffff', directed=False)
        net.force_atlas_2based()
        
        # Add nodes
        for node in self.network.nodes():
            categories = self.word_meanings[node]  # Get categories for the node
            category_str = ', '.join(sorted(categories))
            cluster_id = self.word_to_cluster.get(node, -1)  # Get cluster information
            color = '#ff8c8c' if node == 'animal' else f'hsl({hash(str(cluster_id)) % 360}, 70%, 70%)'  # Color nodes by cluster
            net.add_node(node, 
                        label=node,
                        title=f"{node}\nCategories: {category_str}\nCluster: {cluster_id}",
                        color=color,
                        size=20,
                        font={'size': 20})
        
        # Add edges
        for edge in self.network.edges(data=True):
            source, target, data = edge
            weight = float(data.get('weight', 1.0))
            width = weight * 2
            net.add_edge(source, 
                        target, 
                        value=width,
                        title=f"Similarity: {weight:.2f}",
                        label=f"{weight:.2f}",
                        font={'size': 10})
        
        net.set_options("""
        var options = {
          "nodes": {
            "font": {"size": 20}
          },
          "edges": {
            "font": {"size": 10},
            "smooth": {
              "type": "continuous",
              "forceDirection": "none"
            }
          },
          "physics": {
            "forceAtlas2Based": {
              "springLength": 200,
              "springConstant": 0.05,
              "damping": 0.4
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }
        """)
        
        net.save_graph(filename)
        print(f"Interactive plot saved to {filename}")


    def get_network_stats(self) -> Dict:
        """Get basic statistics about the network"""
        return {
            'num_nodes': self.network.number_of_nodes(),
            'num_edges': self.network.number_of_edges(),
            'average_degree': np.mean([d for n, d in self.network.degree()]),
            'clustering_coefficient': nx.average_clustering(self.network),
            'connected_components': nx.number_connected_components(self.network),
            'num_clusters': len(self.clusters)
        }


    def squash_edge_weights(self, alpha: float) -> None:
        """
        Adjust edge weights to be closer to 0.5 and save the updated plot.
        - alpha: float, the compression factor (0 = no adjustment, 1 = all weights become 0.5)
        """
        if not (0 <= alpha <= 1):
            raise ValueError("Alpha must be in the range [0, 1]")
        for u, v, data in self.network.edges(data=True):
            original_weight = data['weight']
            squashed_weight = (1 - alpha) * original_weight + alpha * 0.5
            self.network[u][v]['weight'] = squashed_weight



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build semantic network")
    # 'results/category_corpus.txt' or 'results/llm_corpus.txt'
    parser.add_argument("--corpus_file", type=str, required=True)
    args = parser.parse_args()

    if "category" in args.corpus_file:
        learner = SemanticNetworkLearner(rho=0.6, rho_animal=0.3)
    elif "llm" in args.corpus_file:
        learner = SemanticNetworkLearner(rho=0.4, rho_animal=0.2)
    else:
        learner = SemanticNetworkLearner(rho=0.8, rho_animal=0.4)
    
    learner.process_corpus_file(args.corpus_file)
    learner.squash_edge_weights(alpha=0.7)

    if 'category' in args.corpus_file:
        learner.plot_network('results/networks/category_network_plot.html')
    elif 'llm' in args.corpus_file:
        learner.plot_network('results/networks/llm_network_plot.html')
    else:
        learner.plot_network('results/networks/network_plot.html')
    
    stats = learner.get_network_stats()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
