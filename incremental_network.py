import numpy as np
from collections import defaultdict
import networkx as nx
from typing import Dict, List, Set
import re
from pyvis.network import Network


class SemanticNetworkLearner:
    def __init__(self, rho: float = 0.8, rho_animal: float = 0.4):
        # Thresholds for edge creation
        self.rho = rho  # Regular similarity threshold
        self.rho_animal = rho_animal  # Threshold for connections to 'animal' node
        self.word_meanings: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.network = nx.Graph()
        self.all_features: Set[str] = set()
        self.clusters: Dict[int, Set[str]] = {}
        self.cluster_prototypes: Dict[int, Dict[str, float]] = {}
    

    def process_corpus_file(self, filename: str) -> None:
        """Process the corpus file"""
        cur_sentence, cur_features = [], []
        
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('SENTENCE:'):
                    cur_sentence = line.replace('SENTENCE:', '').strip().split()
                elif line.startswith('SEM_REP:'):
                    cur_features = [f for f in line.replace('SEM_REP:', '').strip().split(',') if f]
                    # Process the utterance-scene pair
                    if cur_sentence and cur_features:
                        self.process_utterance(cur_sentence, cur_features)
    

    def process_utterance(self, utterance: List[str], features: List[str]) -> None:
        """Process a single utterance-scene pair"""
        self.all_features.update(features)
        
        # Simple alignment: associate all features with all content words
        for word in utterance:
            if re.match(r'^[a-z]+$', word):  # Simple content word check
                # Update word meaning probabilities
                total_weight = len(features)
                for feature in features:
                    self.word_meanings[word][feature] += 1.0 / total_weight
                
                # Normalize probabilities
                total = sum(self.word_meanings[word].values())
                for feature in self.word_meanings[word]:
                    self.word_meanings[word][feature] /= total
                
                # Update network
                self._update_network(word)
    

    def _cosine_similarity(self, word1: str, word2: str) -> float:
        """Calculate cosine similarity between two words"""
        vec1 = np.array([self.word_meanings[word1].get(f, 0) for f in self.all_features])
        vec2 = np.array([self.word_meanings[word2].get(f, 0) for f in self.all_features])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return np.dot(vec1, vec2) / (norm1 * norm2)
    

    def _update_network(self, updated_word: str) -> None:
        """Update network connections for a word"""
        # Add node if it doesn't exist
        if updated_word not in self.network:
            self.network.add_node(updated_word)
            
        # Update existing connections
        for neighbor in list(self.network.neighbors(updated_word)):
            sim = self._cosine_similarity(updated_word, neighbor)
            threshold = self.rho_animal if 'animal' in (updated_word, neighbor) else self.rho
            if sim >= threshold:
                self.network[updated_word][neighbor]['weight'] = sim
            else:
                self.network.remove_edge(updated_word, neighbor)
        
        # Check for new connections with existing nodes
        for word in self.network.nodes():
            if word != updated_word and not self.network.has_edge(updated_word, word):
                sim = self._cosine_similarity(updated_word, word)
                threshold = self.rho_animal if 'animal' in (updated_word, word) else self.rho
                
                if sim >= threshold:
                    self.network.add_edge(updated_word, word, weight=sim)
    
    
    def plot_network(self, filename: str) -> None:
        """Create an interactive HTML visualization of the semantic network with labels"""
        net = Network(height='750px', width='100%', bgcolor='#ffffff', directed=False)
        net.force_atlas_2based()
        
        for node in self.network.nodes():
            color = '#ff8c8c' if node == 'animal' else '#97c2fc'
            net.add_node(node, 
                        label=node,
                        title=node,
                        color=color,
                        size=20,
                        font={'size': 20})
        
        for edge in self.network.edges(data=True):
            source, target, data = edge
            weight = float(data.get('weight', 1.0))
            width = weight * 2
            net.add_edge(source, 
                        target, 
                        value=width,
                        title=f"Weight: {weight:.2f}",  # Show on hover
                        label=f"{weight:.2f}",  # Show weight on edge
                        font={'size': 10})  # Size of edge label
        
        net.set_options("""
        var options = {
        "nodes": {
            "font": {
            "size": 20
            }
        },
        "edges": {
            "font": {
            "size": 10
            },
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
            'connected_components': nx.number_connected_components(self.network)
        }


if __name__ == "__main__":
    learner = SemanticNetworkLearner(rho=0.8, rho_animal=0.4)
    learner.process_corpus_file('learning_corpus.txt')
    learner.plot_network('network_plot.html')
    
    stats = learner.get_network_stats()
    print("\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
