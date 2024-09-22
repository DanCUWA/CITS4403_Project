import random
import time
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
# The Person class acts as the primary Agent for 
# our scenario.
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
    
    def get_health(self): 
        return self.health
    
    def get_diseases(self):
        if len(self.diseases) == 0: 
            return None
        return self.diseases

    def is_sick(self):
        return len(self.diseases) > 0
    # Adding a disease doesn't trigger its' effects immediately, but 
    #   will trigger at the start of each timestep.
    def add_disease(self,disease): 
        self.diseases.append(disease)
    
    def clear_diseases(self): 
        self.diseases = list()
        
# The Disease class acts as the secondary Agent for our scenario.
class Disease:
    def __init__(self): 
        self.infectivity = random.random()
        self.mortality = random.random()
    def infect_person(self,map): 
        map.get_random_person().add_disease(self)

    def will_spread(self): 
        return random.random() < self.infectivity


## The map acts as the environment for our scenario.
class Map:
    def __init__(self,size,k,prob_nurse,prob_person): 
        self.size = size
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        self.num_clusters = k
        self.board = self.generate_array()
        self.cluster_props = self.assign_cluster_probabilities()
        self.place_clusters()

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
                    self.populate_cluster(x_start, y_start, region_size)
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
                if random.random() < self.prob_person:
                    self.add_person(i, j)

    def add_person(self, i, j):
        """Add a person to the grid."""
        if self.board[i][j] is None:
            self.board[i][j] = Person(prob_nurse=self.prob_nurse)

    def get_random_person(self): 
        return random.choice([person for row in self.board for person in row if person is not None and not person.is_nurse()])

    def get_element_at(self,i,j): 
        if self.board[i][j] is None: 
            # return "Empty"
            return 0
        if self.board[i][j].is_nurse():
            # return "Nurse" 
            return 2
        elif self.board[i][j].is_sick(): 
            return 3
        else: 
            # return "Person"
            return 1
    
    # Assumes caller is sick.
    def infect_surrounding(self,i,j): 
        cur_diseases = self.board[i][j].get_diseases()
        num = 0
        for neighbour in self.get_neighbours(i,j):
            if neighbour is not None: 
                for disease in cur_diseases: 
                    if disease.will_spread():
                        neighbour.add_disease(disease)
        
    def get_neighbours(self,i,j):
        neighbours =  list()
        for i in range(i-1,i+2):
            for j in range(j-1,j+2): 
                try:
                    neighbours.append(self.board[i][j])
                except Exception as e: 
                    print("Invalid",e)
        return neighbours

    def get_infected_surrounding(self,i,j):
        count = 0
        for neighbour in self.get_neighbours(i,j): 
            if neighbour is not None and neighbour.is_sick():
                count += 1
        return count

    def get_safest_surrounding(self,start_i,start_j):
        # Can only have a maximum of 9 neighbours.
        lowest = 10
        lowest_vals = None
        for i in range(start_i-1,start_i+2):
            for j in range(start_j-1,start_j+2): 
                # Only consider empty neighbours
                if self.get_element_at(i,j) == 0:
                    unsafe_count = self.get_infected_surrounding(i,j) 
                    print(unsafe_count,"sick around",i,j)
                    if unsafe_count < lowest: 
                        lowest = unsafe_count
                        lowest_vals = [i,j]
        print("Safest square is", lowest_vals, "with ",lowest,"infected surrounding")
        return lowest_vals

    def get_map(self):
        return self.board
    
    def move_to(self,old_i,old_j,new_i,new_j): 
        element = self.board[old_i][old_j]
        self.clear_square(old_i,old_j)
        self.board[new_i][new_j] = element

    def get_most_infected_neighbour(self,i,j): 
        lowest_health = 10
        lowest_diseases = 10
        highest_priority = None
        for neighbour in self.get_neighbours(i,j): 
            if neighbour is None: 
                continue
            health = neighbour.get_health()
            diseases = neighbour.get_diseases()
            if diseases is None:
                continue
            diseases = len(diseases)
            if health <= lowest_health and diseases > lowest_diseases: 
                lowest_health = health
                lowest_diseases = diseases
                highest_priority = neighbour
        return highest_priority

    def clear_square(self,i,j):
        self.board[i][j] = None

    def map_to_ints(self): 
        grid = np.zeros((self.size, self.size))
        for i in range(self.size): 
            for j in range(self.size): 
                grid[i][j] = self.get_element_at(i,j)
        return grid
    
    def get_closest_nurse(self,i,j): 
        arr = np.array(self.map_to_ints())
        nurses = np.argwhere(arr == 2)
        # print(nurses)
        min_distance = np.inf
        ret = None
        for coords in nurses: 
            x = coords[0]
            y = coords[1]
            dist = self.get_distance_bewteen(i,j,x,y) 
            if dist < min_distance: 
                min_distance = dist
                ret = [x,y]
            # print("distances",i,j,x,y,dist)
        return ret

    def get_distance_bewteen(self,i,j,i2,j2): 
        return np.sqrt((i - i2)**2 + (j - j2)**2)

    def get_best_move_from_to(self,srci,srcj,desti,destj): 
        min_dist = np.inf
        best_move = None
        for i in range(srci-1,srci+2):
            for j in range(srcj-1,srcj+2): 
                # Can only move to empty square
                if self.get_element_at(i,j) == 0: 
                    if self.get_distance_bewteen(i,j,desti,destj) < min_dist: 
                        best_move = [i,j]
        return best_move

                
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
        for i in range(self.board_size): 
            for j in range(self.board_size): 
                match self.map.get_element_at(i,j): 
                    case 0: 
                        # Empty
                        continue
                    case 1:  
                        # Just Person
                        # num_infected = self.map.get_infected_surrounding(i,j) 
                        # if num_infected == 0: 
                        #     continue
                        # ## Move to the safest empty square 
                        # safest_square = self.map.get_safest_surrounding(i,j)
                        # if safest_square is not None: 
                        #     print("Moving to",safest_square[0],safest_square[1])
                        #     self.map.move_to(i,j,safest_square[0],safest_square[1])
                        continue
                    case 2: 
                        # Nurse
                        # endangered_node = self.map.get_most_infected_neighbour(i,j)
                        # if endangered_node is not None: 
                        #     endangered_node.clear_diseases()
                        continue
                    case 3: 
                        # Person with diseases
                        # self.map.infect_surrounding(i,j)
                        nurse_coords = self.map.get_closest_nurse(i,j)
                        if nurse_coords is None: 
                            # Could just do random move 
                            continue
                        best_move = self.map.get_best_move_from_to(i,j,nurse_coords[0],nurse_coords[1])
                        print("Infected at",i,j,"optimal move is",best_move,"to nurse at",nurse_coords)
                        self.map.move_to(i,j,best_move[0],best_move[1])
                        continue
    
    def view(self):
        """Visualize the grid using matplotlib."""
        grid = np.zeros((self.board_size, self.board_size))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                grid[i][j] = self.map.get_element_at(i,j)
        
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='0 = Empty, 1 = Person, 2 = Nurse')
        plt.title("Randomly Placed Clusters with People and Nurses")
        plt.show()

    def show_map(self): 
        return self.map.get_map()