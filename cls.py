import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from queue import Queue
import copy
from enum import Enum
import uuid
# 
class Condition(Enum): 
    NO_INFECTED = 0
    NO_NURSES = 1
    NO_PEOPLE = 2
    NO_INFECTED_OR_NURSES = 3
# Numerical representation of board state.
class Tile(Enum): 
    EMPTY = 0
    PERSON = 1
    NURSE = 2
    INFECTED = 3
# The Person class acts as the primary Agent for 
# our scenario.
class Person: 
    def __init__(self,prob_nurse): 
        self.nurse = False
        self.immunity = 0
        # Chance to resist disease
        self.natural_resistance = random.uniform(0.0,0.2)
        self.starting_health = random.randint(4,7)
        self.health = self.starting_health
        self.diseases = list()
        if random.random() < prob_nurse: 
            self.nurse = True

    def __str__(self):
        ret = ""
        for k in vars(self): 
            ret += k+"->"+str(vars(self)[k])+" | "
        return ret
    
    def is_nurse(self): 

        return self.nurse
    
    def get_health(self): 
        return self.health
    
    def get_diseases(self):
        # if len(self.diseases) == 0: 
        #     return None
        return self.diseases

    def is_sick(self):
        return len(self.diseases) > 0
    
    def appears_sick(self): 
        return self.is_sick() and not self.is_latent()
    
    def is_latent(self):
        return self.health > 3
    # Adding a disease doesn't trigger its' effects immediately, but 
    #   will trigger at the start of each timestep.
    def add_disease(self,disease): 
        if self.has_disease(disease.name): 
            print(self.diseases)
            return
        self.diseases.append(disease)

    
    def clear_diseases(self): 
        # print(self.diseases)
        self.diseases = list()
        # print(self.diseases)

    def has_disease(self,name): 
        for disease in self.diseases: 
            if name == disease.name: 
                return True 
        return False
    
    def update_health(self): 
        for disease in self.diseases: 
            if self.health == 1: 
                if random.random() + self.natural_resistance < disease.kill_chance: 
                    self.health -= 1
            elif random.random() + self.natural_resistance < disease.mortality: 
                print("Health decreased")
                self.health -= 1 
        return self.health
    
# The Disease class acts as the secondary Agent for our scenario.
class Disease:
    def __init__(self): 
        # Chance to infect others
        self.infectivity = random.random()
        # Chance for disease to decrease health
        self.mortality = random.uniform(0.5,1)
        self.kill_chance = random.uniform(0.2,0.4)
        # Unique identifier to prevent duplicates
        self.name = uuid.uuid4()

    def infect_random(self,map):
        # print(map.get_random_person())
        rand_coords = map.get_random_person()
        map.board[rand_coords[0]][rand_coords[1]].add_disease(copy.deepcopy(self))
        return rand_coords

    def will_spread(self): 
        return random.random() < self.infectivity

    def __str__(self):
        ret = ""
        for k in vars(self): 
            ret += k+"->"+str(vars(self)[k])+" | "
        return ret


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

    def is_nurse_adjacent(self,i,j): 
        neighbours = self.get_neighbours(i,j)
        for neighbour,ni,nj in neighbours:
            if neighbour is not None and neighbour.is_nurse(): 
                return True 
        return False
    
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
        return random.choice([[i,j] for i,row in enumerate(self.board) for j,col in enumerate(row) if self.get_element_at(i,j) == Tile.PERSON])
    
    def check_in_bounds(self,i,j): 
        return i >= 0 and i < self.size and j >= 0 and j < self.size 

    def get_element_at(self,i,j): 
        if self.board[i][j] is None: 
            # return "Empty"
            return Tile.EMPTY
        if self.board[i][j].is_nurse():
            # return "Nurse" 
            return Tile.NURSE
        elif self.board[i][j].is_sick(): 
            return Tile.INFECTED
        else: 
            # return "Person"
            return Tile.PERSON
    
    # Assumes caller is sick.
    def infect_surrounding(self,i,j): 
        if not self.board[i][j].is_sick(): 
            return 
        cur_diseases = self.board[i][j].get_diseases()
        num = 0
        for neighbour, ni, nj in self.get_neighbours(i,j):
            if neighbour is not None: 
                print("Adding from",cur_diseases,self.board[i][j].get_diseases())
                for disease in cur_diseases: 
                    if disease.will_spread():
                        self.board[ni][nj].add_disease(disease)
                        print(disease, "added to", ni,nj)
        
    def get_neighbours(self,srci,srcj):
        neighbours = list()
        for i in range(srci-1,srci+2):
            for j in range(srcj-1,srcj+2): 
                if i == srci and j == srcj:
                    continue
                if self.check_in_bounds(i,j):
                    neighbours.append([self.board[i][j],i,j])
        return neighbours

    def make_random_move(self,i,j):
        options = [x for x in self.get_neighbours(i,j) if x[0] is None]
        # print("Choosing from",options)
        if len(options) == 0:
            return None
        choice = random.choice(options)
        # print("Chose",choice)
        self.move_to(i,j,choice[1],choice[2])
        return [choice[1],choice[2]]

    def get_infected_surrounding(self,i,j):
        count = 0
        for neighbour, ni, nj in self.get_neighbours(i,j): 
            if neighbour is not None and neighbour.appears_sick():
                count += 1
        return count

    def get_safest_surrounding(self,start_i,start_j):
        # Can only have a maximum of 9 neighbours.
        lowest = 10
        lowest_vals = None
        # for i in range(start_i-1,start_i+2):
        #     for j in range(start_j-1,start_j+2): 
        #         if not self.check_in_bounds(i,j):
        #             continue
        for neighbour, ni, nj in self.get_neighbours(start_i,start_j):
            # Only consider empty neighbours
            if self.get_element_at(ni,nj) == Tile.EMPTY:
                unsafe_count = self.get_infected_surrounding(ni,nj) 
                # print(unsafe_count,"sick around",ni,nj)
                if unsafe_count < lowest: 
                    lowest = unsafe_count
                    lowest_vals = [ni,nj]
        # print("Safest square is", lowest_vals, "with ",lowest,"infected surrounding")
        return lowest_vals

    def get_map(self):
        return self.board
    
    def move_to(self,old_i,old_j,new_i,new_j): 
        element = self.board[old_i][old_j]
        self.clear_square(old_i,old_j)
        self.board[new_i][new_j] = element

    def get_most_infected_neighbour(self,i,j): 
        lowest_health = 10
        lowest_diseases = 0
        highest_priority = [None,None,None]
        for neighbour, ni, nj in self.get_neighbours(i,j): 
            if neighbour is None: 
                continue
            health = neighbour.get_health()
            diseases = neighbour.get_diseases()
            if diseases is None:
                continue
            diseases = len(diseases)
            # print("Infected",neighbour)
            if health <= lowest_health and diseases > lowest_diseases: 
                lowest_health = health
                lowest_diseases = diseases
                highest_priority = [neighbour, ni, nj]
        return highest_priority

    def clear_square(self,i,j):
        self.board[i][j] = None

    def map_to_ints(self): 
        grid = np.zeros((self.size, self.size))
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                grid[i][j] = self.get_element_at(i,j).value
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
    
    def get_num_neighbours(self,i,j): 
        count = 0
        for neighbour, ni, nj in self.get_neighbours(i,j):
            if self.get_element_at(ni,nj) != Tile.EMPTY: 
                count += 1
        return count

    def get_distance_bewteen(self,i,j,i2,j2): 
        return np.sqrt((i - i2)**2 + (j - j2)**2)

    def get_best_move_from_to(self,srci,srcj,desti,destj): 
        min_dist = np.inf
        best_move = None
        for i in range(srci-1,srci+2):
            for j in range(srcj-1,srcj+2): 
                if not self.check_in_bounds(i,j):
                    continue
                # Can only move to empty square
                if self.get_element_at(i,j) == Tile.EMPTY: 
                    print(i,j,desti,destj,"distance ",self.get_distance_bewteen(i,j,desti,destj))
                    dist = self.get_distance_bewteen(i,j,desti,destj)
                    if dist < min_dist: 
                        best_move = [i,j]
                        min_dist = dist
        # print("Best move is",best_move, "with distance", )
        return best_move

    def copy(self): 
        return copy.deepcopy(self)

    def get_total_people(self):
        count = 0
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                if self.get_element_at(i,j) != Tile.EMPTY:
                    count += 1
        return count   
    
    def check_end(self):
        any_infected = False
        any_nurses = False
        any_people = False
        grid = np.zeros((self.size, self.size))
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                if self.get_element_at(i,j) == Tile.PERSON:
                    any_people = True
                if self.get_element_at(i,j) == Tile.INFECTED: 
                    any_infected = True
                if self.get_element_at(i,j) == Tile.NURSE:
                    any_nurses = True
        if not any_people: 
            return Condition.NO_PEOPLE
        if not any_infected: 
            return Condition.NO_INFECTED
        # if not any_nurses: 
        #     return Condition.NO_NURSES
        return False

class Simulation: 

    def __init__(self,board_size=100,num_clusters=4,prob_nurse=0.2,prob_person=0.4): 
        self.board_size = board_size
        self.num_clusters = num_clusters
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        self.metrics = SimMetrics()
        # self.initial_hotspots = list()
        self.starting_map = None
        # self.start_count = 0
        self.map = Map(board_size,num_clusters,prob_nurse=prob_nurse,prob_person=prob_person)
        self.disease_choices = list()
        # self.dead = list()
        # self.iterations = 0
        # self.previous = None
        self.running = False
        self.all_metrics = list()

    def get_sim_params(self):
        return [self.board_size,self.num_clusters,self.prob_nurse,self.prob_person]

    def get_diseases(self): 
        return self.disease_choices
            
    def add_disease_option(self): 
        self.disease_choices.append(Disease())

    def start(self): 
        self.metrics.set_start_count(self.map.get_total_people())
        self.add_disease_option()
        # self.add_disease_option()
        for disease in self.disease_choices: 
            num_to_infect = random.randint(1,4)
            print("Infecting",num_to_infect)
            for i in range(0,num_to_infect): 
                self.metrics.add_hotspot(disease.infect_random(self.map))
        # self.starting_map = self.map.copy()
        self.metrics.set_first_map(self.map.copy())

    def step(self): 
        if self.map.check_end() is not False:
            print("Simulation Ended",self.map.check_end()) 
            return

        # self.iterations += 1 
        self.metrics.increment_iterations()
        # self.previous = self.map.copy()
        new_map = self.map.copy()
        indices = [[x,y] for x in range(self.board_size) for y in range(self.board_size)]
        random.shuffle(indices)
        for i,j in indices: 
            if not self.map.check_in_bounds(i,j): 
                continue
            match self.map.get_element_at(i,j): 
                case Tile.EMPTY: 
                    # Empty
                    continue
                case Tile.PERSON:  
                    # Just Person
                    num_infected = new_map.get_infected_surrounding(i,j) 
                    if num_infected == 0: 
                        new_map.make_random_move(i,j)
                        continue
                    ## Move to the safest empty square 
                    safest_square = new_map.get_safest_surrounding(i,j)
                    if safest_square is not None: 
                        # print("Moving to",safest_square[0],safest_square[1])
                        new_map.move_to(i,j,safest_square[0],safest_square[1])
                    continue
                case Tile.NURSE: 
                    # Nurse
                    endangered_node,ni,nj = new_map.get_most_infected_neighbour(i,j)
                    print("Neighbour",i,j,endangered_node)
                    if endangered_node is not None: 
                        self.metrics.increment_healed()
                        new_map.board[ni][nj].clear_diseases()
                    continue
                case Tile.INFECTED: 
                    cur_health = new_map.board[i][j].update_health()
                    print("Node health is", cur_health)
                    if cur_health <= 0: 
                        print("Node died")
                        self.metrics.add_dead(new_map.board[i][j])
                        new_map.board[i][j] = None
                        continue
                    # Person with diseases
                    new_map.infect_surrounding(i,j)
                    # print("Moving infected at",i,j)
                    if self.map.is_nurse_adjacent(i,j):
                        # print("Nurse adjacent to",i,j)
                        continue
                    nurse_coords = new_map.get_closest_nurse(i,j)
                    if nurse_coords is None: 
                        # Could just do random move 
                        new_map.make_random_move(i,j)
                        continue
                    best_move = new_map.get_best_move_from_to(i,j,nurse_coords[0],nurse_coords[1])
                    # print("Infected at",i,j,"optimal move is",best_move,"to nurse at",nurse_coords)
                    new_map.move_to(i,j,best_move[0],best_move[1])
                    continue
        self.map = new_map
        self.all_metrics.append(StepMetrics(self.metrics.copy(),new_map.copy()))
    
    def view(self,chosen_map=None):
        """Visualize the grid using matplotlib."""
        grid = np.zeros((self.board_size, self.board_size))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if chosen_map is None:
                    grid[i][j] = self.map.get_element_at(i,j).value
                else: 
                    grid[i][j] = chosen_map.get_element_at(i,j).value
        
        bounds = [0, 1, 2, 3 ]
        cmap = 'viridis'
        norm = plt.Normalize(vmin=bounds[0], vmax=bounds[-1])

        plt.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        plt.colorbar(label='0 = Empty, 1 = Person, 2 = Nurse, 3 = Infected',cmap=cmap,norm=norm)
        plt.title("Randomly Placed Clusters with People and Nurses")
        plt.show()

    def show_map(self): 
        return self.map.get_map()
    
    def run_to_end(self): 
        while self.map.check_end() is False:
            self.step()

    def end(self): 
        # total_matched, total_possible = self.metrics.get_hotspot_density()
        # print(total_matched,"starting neighbours out of",total_possible,"possible")
        # print("Started with",self.metrics.get_start_count(),"people. Ended with "+str(self.map.get_total_people()) 
        #       + ". " + str(self.metrics.get_start_count() - self.map.get_total_people()),"people died in",self.metrics.iterations,"iterations.")
        # print("The following died:",self.metrics.get_dead())
        # print(f"{self.metrics.get_healed()} healed.")
        print(f"Ended because of: {self.map.check_end()}")
        # print("Started at:")
        # self.view(chosen_map=self.metrics.get_first_map())
        # print("Finished at:")
        # self.view()
        for met in self.all_metrics: 
            metric_cls : SimMetrics = met.metrics 
            it = metric_cls.get_iterations()
            print(it,str(metric_cls))
class StepMetrics(): 
    def __init__(self,sim_metrics,map):
        self.metrics = sim_metrics
        self.metrics.set_first_map(map)

class SimMetrics:
    def __init__(self): 
        self.dead = list()
        self.num_healed = 0
        self.iterations = 0
        self.start_count = 0
        self.initial_hotspots = list()
        self.starting_map = None

    def add_dead(self,person): 
        self.dead.append(person)

    def increment_healed(self):
        self.num_healed += 1

    def increment_iterations(self): 
        self.iterations += 1 

    def add_hotspot(self,hotspot): 
        self.initial_hotspots.append(hotspot)
    
    def get_hotspot_density(self):
        total_possible = 0
        total_matched = 0
        for hotspot in self.initial_hotspots: 
            print(hotspot)
            i = hotspot[0]
            j = hotspot[1]
            total_possible += len(self.starting_map.get_neighbours(i,j))
            total_matched += self.starting_map.get_num_neighbours(i,j)
        return total_matched, total_possible

    def set_first_map(self,map): 
        self.starting_map = map

    def get_first_map(self):
        return self.starting_map
    
    def get_dead(self):
        return self.dead.copy()
    
    def get_healed(self): 
        return self.num_healed
    
    def set_start_count(self,num): 
        self.start_count = num

    def get_iterations(self): 
        return self.iterations
    
    def get_start_count(self): 
        return self.start_count
    
    def copy(self): 
        return copy.deepcopy(self)
    
    def __str__(self):
        ret = ""
        for k in vars(self): 
            ret += k+":"+str(vars(self)[k])+" , "
        return ret