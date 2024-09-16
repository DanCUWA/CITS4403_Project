import random
import pygame
import time
import np
import matplotlib.pyplot as plt
# The Person class acts as the primary Agent for 
#   our scenario.
class Person: 
    def __init__(self,prob_nurse): 
        self.nurse = False
        self.immunity = 0
        self.natural_resistance = random.random()
        self.health = random.randint(4,7)
        self.diseases = list()
        if random.random() < prob_nurse: 
            self.nurse = True
    
    def is_nurse(self): 
        return self.nurse
    
    def get_diseases(self):
        if len(self.diseases) == 0: 
            return None
        return self.diseases

    # Adding a disease doesn't trigger its' effects immediately, but 
    #   will trigger at the start of each timestep.
    def add_disease(self,disease): 
        self.diseases.append(disease)
        
# The Disease class acts as the secondary Agent for our scenario.
class Disease:
    def __init__(self): 
        self.infectivity = random.randint(0,0.5)
        self.mortality = random.randint(0,0.5)
    def infect_person(self,map): 
        map.get_random_person().add_disease(self)


## The map acts as the environment for our scenario.
class Map:
    def __init__(self,size,k,prob_nurse,prob_person): 
        self.size = size
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        self.num_clusters = k
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

    def populate_cluster(self, x_start, y_start, region_size):
        """Populate a region with people, ensuring separation from other clusters."""
        for i in range(x_start, x_start + region_size):
            for j in range(y_start, y_start + region_size):
                if random.random() < self.prob_person:  # 70% chance to place a person in the region
                    self.add_person(i, j)

    def add_person(self, i, j):
        """Add a person to the grid."""
        if self.board[i][j] is None:
            self.board[i][j] = Person(prob_nurse=self.prob_nurse)

    def get_random_person(self): 
        return random.choice([person for row in self.board for person in row if person is not None and not person.is_nurse()])

    def generate_array(self): 
        for i in range(self.size): 
            for j in range(self.size):
                if random.random() < self.prob_person:
                    self.add_person(i,j,[0.2])
                # pass
                #     
    def get_element_at(self,i,j): 
        if self.board[i][j] == 0: 
            return "Empty"
        if self.board[i][j].is_nurse():
            return "Nurse" 
        else: 
            return "Person"
        
    # def add_person(self,i,j,cluster_props):
    #     self.board[i][j] = Person(prob_nurse=self.prob_nurse)

    def get_map(self):
        return self.board

class Simulation: 

    def __init__(self,board_size=100,num_clusters=4,prob_nurse=0.2,prob_person=0.4): 
        self.board_size = board_size
        self.num_clusters = num_clusters
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        self.map = Map(board_size,num_clusters,prob_nurse=prob_nurse,prob_person=prob_person)
        self.disease_choices = list()


    def add_disease_option(self): 
        self.disease_choices.append(Disease())

    def start(self): 
        self.add_disease_option()
        for disease in self.disease_choices: 
            disease.infect_person(self.map)

    def step(self): 
        pass
    
    # def view(self):
    #     pygame.init()
    #     screen_size = 750
    #     size_offset = float(float(screen_size) / float(self.board_size)) * 1.1
    #     print("Offset ", size_offset)
    #     screen = pygame.display.set_mode([screen_size, screen_size])
    #     screen.fill((255, 255, 255))
    #     for i in range(self.board_size): 
    #         for j in range(self.board_size):
    #             x = i * size_offset
    #             y = j * size_offset
    #             el_type = self.map.get_element_at(i,j)
    #             print("(",el_type,end=") , ")
    #             if el_type == "Person": 
    #                 pygame.draw.circle(screen, (0, 0, 255), (x,y), 3)
    #             elif el_type == "Nurse":
    #                 pygame.draw.circle(screen, (0, 255, 0), (x,y), 3)
    #     # while True:
    #     pygame.display.flip()
    #     time.sleep(10)
        # pygame.quit()

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

    def show_map(self): 
        return self.map.get_map()
