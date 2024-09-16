import random
import numpy as np
import matplotlib.pyplot as plt
from cls.py import *

PROB_NURSE = 0.01
NUM_CLUSTERS = 4

class Person: 
    def __init__(self, prob_nurse): 
        self.nurse = False
        self.immunity = 0
        self.natural_resistance = random.random()
        self.health = random.randint(4, 7)
        self.diseases = list()
        if random.random() < prob_nurse: 
            self.nurse = True
    
    def is_nurse(self): 
        return self.nurse
    
    def get_diseases(self):
        if len(self.diseases) == 0: 
            return None
        return self.diseases

    def add_disease(self, disease): 
        self.diseases.append(disease)
        
    
class Disease:
    def __init__(self): 
        self.infectivity = random.uniform(0, 0.5)
        self.mortality = random.uniform(0, 0.5)

    def infect_person(self, map_obj): 
        map_obj.get_random_person().add_disease(self)

class Map:
    def __init__(self, size, num_clusters): 
        self.size = size
        self.num_clusters = num_clusters
        self.board = self.generate_array()
        self.cluster_props = self.assign_cluster_probabilities()

    def generate_array(self): 
        # Initialize the board with None (empty spaces)
        return [[None for _ in range(self.size)] for _ in range(self.size)]

    def assign_cluster_probabilities(self):
        # Assign different probabilities to different clusters
        return [random.uniform(0.1, 0.5) for _ in range(self.num_clusters)]

    def place_clusters(self):
        # Randomly place clusters on the grid while ensuring they don't overlap and are separated by at least 1 space
        cluster_areas = []
        max_attempts = 1000  # Maximum number of attempts to place clusters without overlap
        
        for cluster_index in range(self.num_clusters):
            placed = False
            attempts = 0
            region_size = random.randint(10, 20)  # Random size for each cluster
            
            while not placed and attempts < max_attempts:
                attempts += 1
                # Random top-left corner for the cluster
                x_start = random.randint(0, self.size - region_size - 1)
                y_start = random.randint(0, self.size - region_size - 1)

                # Check if the cluster can fit without touching others (minimum 1 space around it)
                if self.is_region_free(x_start, y_start, region_size):
                    cluster_areas.append((x_start, y_start, region_size))
                    prob_nurse = self.cluster_props[cluster_index]
                    self.populate_cluster(x_start, y_start, region_size, prob_nurse)
                    placed = True
        
        if len(cluster_areas) < self.num_clusters:
            print("Warning: Could not place all clusters due to space constraints.")

    def is_region_free(self, x_start, y_start, region_size):
        """Check if a region is free and has at least one space buffer around."""
        # Check if the region is free and has at least one space around
        for i in range(x_start - 1, x_start + region_size + 1):
            for j in range(y_start - 1, y_start + region_size + 1):
                if i < 0 or i >= self.size or j < 0 or j >= self.size:
                    continue  # Ignore out-of-bounds checks
                if self.board[i][j] is not None:  # If there is a person or nurse already
                    return False
        return True

    def populate_cluster(self, x_start, y_start, region_size, prob_nurse):
        """Populate a region with people, ensuring separation from other clusters."""
        for i in range(x_start, x_start + region_size):
            for j in range(y_start, y_start + region_size):
                if random.random() < 0.7:  # 70% chance to place a person in the region
                    self.add_person(i, j, prob_nurse)

    def add_person(self, i, j, prob_nurse):
        """Add a person to the grid."""
        if self.board[i][j] is None:
            self.board[i][j] = Person(prob_nurse)

    def display_map(self):
        """Visualize the grid using matplotlib."""
        grid = np.zeros((self.size, self.size))
        
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] is None:
                    grid[i][j] = 0  # Empty space
                elif self.board[i][j].is_nurse():
                    grid[i][j] = 2  # Nurse
                else:
                    grid[i][j] = 1  # Normal person
        
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='0 = Empty, 1 = Person, 2 = Nurse')
        plt.title("Randomly Placed Clusters with People and Nurses")
        plt.show()

    def get_random_person(self): 
        return random.choice([person for row in self.board for person in row if person is not None])

# Test the Map class and display the grid
if __name__ == "__main__":
    size = 100  # Grid size 100x100
    num_clusters = 4  # You can adjust the number of clusters
    map_obj = Map(size, num_clusters)

    # Place clusters randomly with spacing constraints
    map_obj.place_clusters()

    # Display the generated map/grid
    map_obj.display_map()


