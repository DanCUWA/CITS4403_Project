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
import heapq
import pandas as pd



class Condition(Enum): 
    NO_INFECTED = 0
    NO_NURSES = 1
    NO_PEOPLE = 2
    NO_INFECTED_OR_NURSES = 3

class Tile(Enum): 
    EMPTY = 0
    PERSON = 1
    NURSE = 2
    INFECTED = 3
# The Person class acts as the primary Agent for 
# our scenario.
"""
Class representing people in the simulation.
Each person has inherent attributes modelling 
how they interact with the environment. 
"""
class Person: 
    """
    Constructor
    Parameters: 
        float prob_nurse: probability that 
                    a new person is a nurse
    """
    def __init__(self,prob_nurse): 
        self.nurse = False
        # How long until this person (recovering from a disease)
        #   can be reinfected
        self.immunity = 0
        # Chance to resist being infected by a disease
        self.natural_resistance = random.uniform(0.0,0.2)
        # Initial assigned health, retain a copy for metrics
        self.starting_health = random.randint(4,7)
        # Current health, updated throughout simulation
        self.health = self.starting_health
        # List of all diseases individual is currently infected with.
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
        """
        Checks if a given Person has any diseases.
        Returns Boolean
        """
        return len(self.diseases) > 0
    
    def appears_sick(self): 
        """
        Checks if other people can see this Person as sick.
        Accounts for latent period, where others may not know 
        the current Person is infected. 
        """
        return self.is_sick() and not self.is_latent()
    
    def is_latent(self):
        """
        People will only appear as infected to other people 
        when they reach a certain level of sickness. 
        Returns:
            True    If person is in latent period
            False   Otherwise
        """
        return self.health > 3
    # Adding a disease doesn't trigger its' effects immediately, but 
    #   will trigger at the start of each timestep.
    def add_disease(self,disease): 
        """
        Add a new diseases to the current person when they are infected 
        by another person. Adding a disease doesn't trigger its' effects 
        immediately, but will trigger at the start of each timestep.
        Parameters: 
            Diseases disease: An object of the Disease class that the 
                                person has been infected with 
        """
        if self.has_disease(disease.name): 
            print(self.diseases)
            return
        self.diseases.append(disease)

    
    def clear_diseases(self): 
        """
        Add a new diseases to the current person when they are infected 
        by another person. Adding a disease doesn't trigger its' effects 
        immediately, but will trigger at the start of each timestep.
        Parameters: 
            Diseases disease: An object of the Disease class that the 
                                  person has been infected with.
        """
        # print(self.diseases)
        self.diseases = list()
        # print(self.diseases)

    def has_disease(self,name): 
        """
        Checks if a person already has an instance of a disease. Used 
        to ensure no one can get sick from multiple instances of the same 
        disease simultaneously. 
        Parameters: 
            str name: The name of the disease to check.. 
        Returns: 
            Boolean:    True if person is already infected with the relevant 
                            disease.
                        False otherwise.
        """
        for disease in self.diseases: 
            if name == disease.name: 
                return True 
        return False
    
    def update_health(self):
        """
        Runs before an infected person moves in the simulation. 
        Reduces their health according to the properties of the 
        diseases they are infected with.  
        Returns: 
            int health:     The updated health value for the agent. 
        """ 
        for disease in self.diseases: 
            if self.health == 1: 
                if random.random() + self.natural_resistance < disease.kill_chance: 
                    self.health -= 1
            elif random.random() + self.natural_resistance < disease.mortality: 
                print("Health decreased")
                self.health -= 1 
        return self.health
    
# The Disease class acts as the secondary Agent for our scenario.
"""
The Disease class acts as the secondary Agent for our scenario.
To accurately model real world scenarios, we account for multiple 
types of diseases that vary in virulence and lethality. 
"""
class Disease:
    """
    Constructor for Disease Agents
    """
    def __init__(self): 
        # Chance the diesease will spread to others
        self.infectivity = random.random()
        # Chance for disease to decrease an infected Persons' health
        self.mortality = random.uniform(0.5,1)
        # Chance that a disease will actually kill a Person (reduce 
        #   health below 0)
        self.kill_chance = random.uniform(0.2,0.4)
        # Unique identifier to prevent duplicates
        self.name = uuid.uuid4()
    def infect_random(self,map):
        """
            Runs before an infected person moves in the simulation. 
            Reduces their health according to the properties of the 
            diseases they are infected with.  
            Parameters: 
                Map map:    The map to infect a random Person in.    
            Returns: 
                int[] coords:     The coordinates [x,y] of the chosen 
                                    Person. 
        """ 
        # print(map.get_random_person())
        rand_coords = map.get_random_person()
        map.board[rand_coords[0]][rand_coords[1]].add_disease(copy.deepcopy(self))
        return rand_coords

    def will_spread(self): 
        """
            Determines if a disease will spread to a Person, 
            based on the Diseases' infectivity.
            Returns: 
                Boolean:    True if the disease will spread,
                            False otherwise 
        """ 
        return random.random() < self.infectivity

    def __str__(self):
        ret = ""
        for k in vars(self): 
            ret += k+"->"+str(vars(self)[k])+" | "
        return ret


"""
The Map class acts as the environment for our simulation, 
providing the board and other required features. 
"""
class Map:
    """
    Constructor for the Map class.
    Parameters: 
        int size:           The size of the board.
        int k:              The number of clusters to  
                                start with on the board.
        float prob_nurse:   Probability that a new
                                person is a nurse.
        float prob_person:  Probability that an empty 
                                square on the first map
                                is a person.
    """
    def __init__(self,size,k,prob_nurse,prob_person): 
        self.size = size
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        self.num_clusters = k
        # The 2d array that will hold the board state. 
        self.board = self.generate_array()
        # Cluster initalisation 
############# OSCAR ADD COMMENT DESCRIBING WHAT THIS DOES ############
        self.cluster_props = self.assign_cluster_probabilities()
        self.place_clusters()

    def dijkstra(self, start_i, start_j):
        """Use Dijkstra's algorithm to find the shortest path from (start_i, start_j) to all nurses."""
        # Priority queue to hold the cells to be processed
        pq = []
        # Distance dictionary to hold the shortest distance to each cell
        distances = { (i, j): float('inf') for i in range(self.size) for j in range(self.size) }
        # Dictionary to store only the distances to nurses
        nurse_distances = {}
        
        # Initialize the start point's distance
        distances[(start_i, start_j)] = 0
        heapq.heappush(pq, (0, (start_i, start_j)))  # (distance, (i, j))

        while pq:
            current_distance, (current_i, current_j) = heapq.heappop(pq)

            # If this distance is greater than the recorded distance, skip it
            if current_distance > distances[(current_i, current_j)]:
                continue

            # Explore neighbors
            for neighbour, ni, nj in self.get_neighbours(current_i, current_j):
                if self.get_element_at(ni, nj) == Tile.EMPTY:
                    distance = current_distance + 1  # All moves have a cost of 1                    
                    if distance < distances[(ni, nj)]:
                        distances[(ni, nj)] = distance
                        heapq.heappush(pq, (distance, (ni, nj)))
                        
                        # If the neighbor is a nurse, add it to the nurse_distances dictionary
                        if self.get_element_at(ni, nj) == Tile.NURSE:
                            nurse_distances[(ni, nj)] = distance
                if self.get_element_at(ni, nj) == Tile.NURSE:
                    distance = current_distance
                    if distance < distances[(ni, nj)]:
                        distances[(ni, nj)] = distance
                        heapq.heappush(pq, (distance, (ni, nj)))  
                        # If the neighbor is a nurse, add it to the nurse_distances dictionary
                        if self.get_element_at(ni, nj) == Tile.NURSE:
                            nurse_distances[(ni, nj)] = distance
            
        print(nurse_distances)
        return nurse_distances



    def generate_array(self): 
        """
            Initialise an empty 2D array of the correct size. 
            Returns: 
                arr[][]:    2D array initialised to empty squares. 
        """ 
        return [[None for _ in range(self.size)] for _ in range(self.size)]

    def assign_cluster_probabilities(self):
        # Assign different probabilities to different clusters
        return [random.uniform(0.1, 0.5) for _ in range(self.num_clusters)]

    def is_nurse_adjacent(self,i,j): 
        """
            Checks if a given square is adjacent to a nurse,
            and thus within healing range. 
            Parameters:
                int i:      The x coordinate of the square. 
                int j:      The y coordinate of the square. 
            Returns: 
                Boolean:    True if the square is nurse adjacent,
                            False otherwise. 
        """ 
        neighbours = self.get_neighbours(i,j)
        for neighbour,_,_ in neighbours:
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
        """
            Add a new Person to the map, if the specified 
            coordinates are valid.
            Parameters:
                int i:      The x coordinate to add the Person at. 
                int j:      The y coordinate to add the Person at. 
        """ 
        if self.board[i][j] is None:
            self.board[i][j] = Person(prob_nurse=self.prob_nurse)

    def get_random_person(self): 
        """
            Get a random person from the current choices in the map. 
            Returns:
                int[]:      The coordinates of the randomly chosen 
                            person in form [i,j]
        """ 
        return random.choice([[i,j] for i,row in enumerate(self.board) for j,col in enumerate(row) if self.get_element_at(i,j) == Tile.PERSON])
    
    def check_in_bounds(self,i,j): 
        """
            Error checking to ensure that given coordinates are 
            a valid entry into the board. 
            Parameters:
                int i:      The x coordinate to add the Person at. 
                int j:      The y coordinate to add the Person at. 
            Returns: 
                Boolean:    True if the parameters are a valid coordinate
                                on the map. 
                            False otherwise. 
        """ 
        return i >= 0 and i < self.size and j >= 0 and j < self.size 

    def get_element_at(self,i,j): 
        """
            Returns a numerical value corresponding to the type of 
            Agent present at the given location in the map. 
            Parameters:
                int i:      The x coordinate to add the Person to. 
                int j:      The y coordinate to add the Person to. 
            Returns: 
                Condition:  The enum corresponding to the desired 
                                square. 
        """ 
        if self.board[i][j] is None: 
            return Tile.EMPTY
        elif self.board[i][j].is_nurse():
            return Tile.NURSE
        elif self.board[i][j].is_sick(): 
            return Tile.INFECTED
        else: 
            return Tile.PERSON
    
    def infect_surrounding(self,i,j): 
        """
            Tries to infect all People surrounding an infected individual 
            at the given coordinates. 
            Parameters:
                int i:      The x coordinate of the infected indiividual. 
                int j:      The y coordinate of the infected indiividual. 
        """ 
        if not self.board[i][j].is_sick(): 
            return 
        cur_diseases = self.board[i][j].get_diseases()
        for neighbour, ni, nj in self.get_neighbours(i,j):
            if neighbour is not None: 
                print("Adding from",cur_diseases,self.board[i][j].get_diseases())
                for disease in cur_diseases: 
                    # Calculate if the Person will be infected,
                    #   according to their resistances and the 
                    #   diseases infectivity.
                    if disease.will_spread():
                        self.board[ni][nj].add_disease(disease)
                        print(disease, "added to", ni,nj)
        
    def get_neighbours(self,srci,srcj):
        """
            Returns all elements adjacent to the neighbour 
            at the given coordinates.
            Parameters:
                int srci:           The x coordinate of the source square. 
                int srcj:           The y coordinate of the source square. 
            Returns: 
                [[el,i,j]...]:   A 2D array containing [element,x_coord,y_coord]
                                        for all neighbours of the source square.
        """ 
        neighbours = list()
        for i in range(srci-1,srci+2):
            for j in range(srcj-1,srcj+2): 
                if i == srci and j == srcj:
                    continue
                if self.check_in_bounds(i,j):
                    neighbours.append([self.board[i][j],i,j])
        return neighbours

    def make_random_move(self,i,j):
        """
            Moves an element to a randomly selected 
            valid (empty) adjacent square.
            Parameters:
                int i:           The x coordinate of the element to move. 
                int j:           The y coordinate of the element to move. 
            Returns: 
                None:            If no valid square is adjacent. 
                int[2]:          Coordinates of square that was chosen 
                                    to move to.
        """ 
        options = [x for x in self.get_neighbours(i,j) if x[0] is None]
        # print("Choosing from",options)
        if len(options) == 0:
            return None
        choice = random.choice(options)
        # print("Chose",choice)
        self.move_to(i,j,choice[1],choice[2])
        return [choice[1],choice[2]]

    def get_infected_surrounding(self,i,j):
        """
            Determine how many infected neighbours a Person can see 
            adjacent to a chosen square. 
            Parameters:
                int i:           The x coordinate of the input square. 
                int j:           The y coordinate of the input square. 
            Returns: 
                int count:       The number of infected neighbours seen. 
        """ 
        count = 0
        for neighbour, _, _ in self.get_neighbours(i,j): 
            if neighbour is not None and neighbour.appears_sick():
                count += 1
        return count

    def get_safest_surrounding(self,start_i,start_j):
        """
            Determine which adjacent square is the safest (has the least adjacent 
            infected neighbours) for an uninfected Person to move to. 
            Parameters:
                int start_i:           The x coordinate of the input square. 
                int start_j:           The y coordinate of the input square. 
            Returns: 
                int[2]:                The [i,j] coordinates of the safest square. 
        """ 
        # Can only have a maximum of 9 neighbours.
        lowest = 10
        lowest_vals = None
        for _, ni, nj in self.get_neighbours(start_i,start_j):
            # Only consider empty neighbours
            if self.get_element_at(ni,nj) == Tile.EMPTY:
                unsafe_count = self.get_infected_surrounding(ni,nj) 
                if unsafe_count < lowest: 
                    lowest = unsafe_count
                    lowest_vals = [ni,nj]
        # print("Safest square is", lowest_vals, "with ",lowest,"infected surrounding")
        return lowest_vals

    def get_map(self):
        """
            Returns a copy of the current instance of the board. 
            Returns: 
                int[size][size]:    The current board. 
        """ 
        return self.board
    
    def move_to(self,old_i,old_j,new_i,new_j): 
        """
            Moves an element to a new location. Integrity of new position should
            be verified before call - board[new_i][new_j] should be empty. 
            Parameters:
                int old_i:           The x coordinate of the elements original position. 
                int old_j:           The y coordinate of the elements original position. 
                int new_i:           The x coordinate of the elements new position. 
                int new_j:           The y coordinate of the elements new position. 
        """ 
        element = self.board[old_i][old_j]
        self.clear_square(old_i,old_j)
        self.board[new_i][new_j] = element

    def get_most_infected_neighbour(self,i,j): 
        """
            Determine which square adjacent to the input square is the 
            most at risk, and thus most important for the nurses to heal.
            Parameters:
                int i:                  The x coordinate of the desired square. 
                int j:                  The y coordinate of the desired square. 
            Returns: 
                [neighbour,ni,nj]:      The details of the most at risk square -
                                            the object and its' x, y coordinates.
        """ 
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
        """
            Removes an element from the board.
            Parameters:
                int i:                  The x coordinate of the square to clear. 
                int j:                  The y coordinate of the square to clear. 
        """ 
        self.board[i][j] = None

    def map_to_ints(self): 
        """
            Returns a new board, with all elements converted to their 
            respective numerical values. 
            Returns:
                int[size][size]:    The array containing numerically 
                                        mapped values.
        """ 
        grid = np.zeros((self.size, self.size))
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                grid[i][j] = self.get_element_at(i,j).value
        return grid
    
    def get_closest_nurse(self,i,j): 
        """
            Calculates the coordinates of the closest nurse to the input square.
            Parameters:
                int i:                  The x coordinate of the square to evaluate. 
                int j:                  The y coordinate of the square to evaluate. 
            Returns:
                int[2]:                 The [x,y] coordinates of the closest nurse.
        """ 
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
        """
            Calculates how many people are adjacent to a given square.
            Parameters:
                int i:                  The x coordinate of the square to evaluate. 
                int j:                  The y coordinate of the square to evaluate. 
            Returns:
                int count:              The number of people adjacent to the square. 
        """ 
        count = 0
        for _, ni, nj in self.get_neighbours(i,j):
            if self.get_element_at(ni,nj) != Tile.EMPTY: 
                count += 1
        return count

    def get_distance_bewteen(self,i,j,i2,j2): 
        """
            Calculates how many people are adjacent to a given square.
            Parameters:
                int i:                  The x coordinate of the source square. 
                int j:                  The y coordinate of the source square. 
                int i2:                 The x coordinate of the destination square. 
                int j2:                 The y coordinate of the destination square. 
            Returns:
                float distance:         The distance between the two squares. 
        """ 
        return np.sqrt((i - i2)**2 + (j - j2)**2)

    def get_best_move_from_to(self,srci,srcj,desti,destj): 
        """
            Greedily calculates the best move to make to get to a desired square
            given the current map state.
            Parameters:
                int srci:                  The x coordinate of the source square. 
                int srcj:                  The y coordinate of the source square. 
                int desti:                 The x coordinate of the destination square. 
                int destj:                 The y coordinate of the destination square. 
            Returns:
                int[2]:                    The coordinates of the best move to the detination.  
        """ 
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
        """
        Returns a deep copy of the current map to facilitate 
        storing past state instances.
        Returns:
            Map map:    A deep copy of the current map.
        """
        return copy.deepcopy(self)

    def get_total_people(self):
        """
        Gets the total number of people in current board. 
        Returns:
            int count:   The number of people in the board. 
        """
        count = 0
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                if self.get_element_at(i,j) != Tile.EMPTY:
                    count += 1
        return count   
    
    def check_end(self):
        """
        Determines if the current state of the map meets 
        any of the simulation termination criteria.
        Returns:
            Condition termination:      The Condition causing the termination. 
            Boolean False:              If no Condition is matched. 
        """
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
                
    def get_average_alive_stats(self): 
        healths = []
        res = []
        for i in range(self.size): 
            for j in range(self.size): 
                if not self.check_in_bounds(i,j):
                    continue
                elif self.get_element_at(i,j) != Tile.EMPTY: 
                    cur_person : Person = self.board[i][j]
                    healths.append(cur_person.starting_health)
                    res.append(cur_person.natural_resistance)
        return np.mean(healths), np.mean(res)
"""
The class responsible for incorporating the previous elements 
and running a simulation instance.
"""
class Simulation: 
    """
    Constructor for the Simulation class.
    Parameters: 
        int board_size:           The size of the board. 
                                  Default 100. 
        int num_clusters:         The number of clusters to start with
                                    on the board.
                                  Default 4
        float prob_nurse:         Probability that a new
                                    person is a nurse.
                                  Default 0.2 
        float prob_person:        Probability that an empty square on 
                                    the first map is a person.
                                  Default 0.4
    """
    def __init__(self,board_size=100,num_clusters=4,prob_nurse=0.2,prob_person=0.4): 
        self.board_size = board_size
        self.num_clusters = num_clusters
        self.prob_nurse = prob_nurse
        self.prob_person = prob_person
        # Hold simulation-wide metrics
        self.metrics = SimMetrics()
        # Copy of the first map 
        self.starting_map = None
        # The current map 
        self.map = Map(board_size,num_clusters,prob_nurse=prob_nurse,prob_person=prob_person)
        # The list of possible diseases for the current simulation
        self.disease_choices = list()
        # Whether or not the simulation has terminated
        self.running = False
        # Holds step-specific metrics for all steps. 
        self.all_metrics = list()
        self.stats = [["infected","not infected", "time step"]]
        self.new_list = list()
        self.old_list = list()
        # self.infected_count = 0

    def get_sim_params(self):
        """ 
            Retrieves the initial parameters used to run the simulation. 
            Returns: 
                arr[4]:     Initial parameters
        """
        return [self.board_size,self.num_clusters,self.prob_nurse,self.prob_person]

    def get_diseases(self): 
        """ 
            Retrieves the list of diseases available to the current simulation. 
            Returns: 
                Disease[]:   List of available diseases
        """
        return self.disease_choices
            
    def add_disease_option(self): 
        """ 
            Creates a new disease, and makes it available 
            to the simulation.
        """
        self.disease_choices.append(Disease())

    def start(self): 
        """ 
            Starts a simulation instance. 
        """
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
        """ 
            Steps through an iteration of the simulation. 
            Squares are processed in a random order.
            Processing involves different actions depending 
            on which element is in the square. 
            If conditions for termination are met, displays 
            the final simulation statistics.  
        """
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
                    
                    distances = new_map.dijkstra(i, j)
                    # Find the closest nurse position
                    closest_nurse_coords = None
                    min_distance = float('inf')
                    for coords, distance in distances.items():
                        if new_map.get_element_at(coords[0], coords[1]) == Tile.NURSE and distance < min_distance:
                            min_distance = distance
                            closest_nurse_coords = coords
                    nurse_coords = new_map.get_closest_nurse(i,j)

                    if closest_nurse_coords is not None:
                        # Move towards the closest nurse
                        best_move = new_map.get_best_move_from_to(i, j, closest_nurse_coords[0], closest_nurse_coords[1])
                        new_map.move_to(i, j, best_move[0], best_move[1])
                        new_map.infect_surrounding(best_move[0],best_move[1])
                        continue
                    if nurse_coords is None: 
                        # Could just do random move 
                        new_map.make_random_move(i,j)
                        continue
                    best_move = new_map.get_best_move_from_to(i,j,nurse_coords[0],nurse_coords[1])
                    # print("Infected at",i,j,"optimal move is",best_move,"to nurse at",nurse_coords)
                    try:
                        new_map.move_to(i,j,best_move[0],best_move[1])
                    except TypeError:
                        print("Infected patient could not move since hospital/nurse met patient capacity.")
                        continue
                    continue
        self.map = new_map
        self.all_metrics.append(StepMetrics(self.metrics.copy(), new_map.copy()))

    def view(self,chosen_map=None):
        """Visualize the grid using matplotlib."""
        grid = np.zeros((self.board_size, self.board_size))
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if chosen_map is None:
                    grid[i][j] = self.map.get_element_at(i,j).value
                else: 
                    grid[i][j] = chosen_map.get_element_at(i,j).value
        
        bounds = [0, 1, 2, 3]
        cmap = 'viridis'
        norm = plt.Normalize(vmin=bounds[0], vmax=bounds[-1])

        plt.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
        plt.colorbar(label='0 = Empty, 1 = Person, 2 = Nurse, 3 = Infected',cmap=cmap,norm=norm)
        plt.title("Randomly Placed Clusters with People and Nurses")
        plt.show()

    def show_map(self): 
        """
            Gets the board of the current map in the simulation. 
            Returns:
                Map map:    The current map
        """
        return self.map.get_map()        

############# OSCAR CHECK COMMENT DESCRIBING WHAT THIS DOES ############
    def run_to_end(self, max_steps=100): 
        """
            Automatically runs the simulation either until it ends, 
            or an assigned number of steps is reached, whichever is 
            first.
            Parameters:
                int max_steps:              The maximum number of steps to execute
            Returns: 
                DataFrame infected_df:      Statistics on infected individuals. 
                DataFrame overall_df:       Statistics on the whole simulation. 
        """
        infected_statistics = []
        infected_stats = self.track_infected_statistics()
        overall_stats = self.track_overall_statistics()
        n = self.get_infected_count()
        if n > 1:
            for row in range(self.get_infected_count()):
                infected_statistics.append(infected_stats[row])
        else:
            infected_statistics.append(infected_stats)
        overall_statistics = [self.track_overall_statistics()]
        current_step = 0

        while self.map.check_end() is False and current_step < max_steps:
            self.step()
            self.view()

            # Track stats for each time step
            infected_stats = self.track_infected_statistics()
            overall_stats = self.track_overall_statistics()

            # Append stats to the list to store them
            n = self.get_infected_count()
            if n > 1:
                for row in range(self.get_infected_count()):
                    infected_statistics.append(infected_stats[row])
            else:
                infected_statistics.append(infected_stats)
            overall_statistics.append(overall_stats)

            current_step += 1  # Increment the step count

        # Once the simulation is done or max steps reached, return the collected statistics
        infected_df = pd.DataFrame(infected_statistics, columns=["Infected X", "Infected Y","Uninfected Count Near", "Total Squares Visited", "Time Step", ])

        overall_df = pd.DataFrame(overall_statistics, columns=["Time Step","Current Infected", "Total Deaths", "Total Healed","Average Alive Health",
                                                                "Average Alive Resistance", "Average Dead Health","Average Dead Resistance"])

        return infected_df, overall_df

    def end(self): 
        """
            Display final statistics upon simulation termination. 
        """       
        # total_matched, total_possible = self.metrics.get_hotspot_density()
        # print(total_matched,"starting neighbours out of",total_possible,"possible")
        # print("Started with",self.metrics.get_start_count(),"people. Ended with "+str(self.map.get_total_people()) 
        #       + ". " + str(self.metrics.get_start_count() - self.map.get_total_people()),"people died in",self.metrics.iterations,"iterations.")
        # print("The following died:",self.metrics.get_dead())
        # print(f"{self.metrics.get_healed()} healed.")
        # print(f"Ended because of: {self.map.check_end()}")
        # # print("Started at:")
        # # self.view(chosen_map=self.metrics.get_first_map())
        # # print("Finished at:")
        # # self.view()
        # for met in self.all_metrics: 
        #     metric_cls : SimMetrics = met.metrics 
        #     it = metric_cls.get_iterations()
        #     print(it,str(metric_cls))
        description,total_matched, total_possible = self.metrics.get_hotspot_density()
        print(description,total_matched,"starting neighbours out of",total_possible,"possible")
        print("Started with",self.metrics.get_start_count(),"people. Ended with "+str(self.map.get_total_people()) 
              + ". " + str(self.metrics.get_start_count() - self.map.get_total_people()),"people died in",self.metrics.iterations,"iterations.")
        print("The following died:",self.metrics.get_dead())
        print(f"{self.metrics.get_healed()} healed.")
        print(f"Ended because of: {self.map.check_end()}")
        for i in self.all_metrics: 
            print("")
        print("Started at:")
        self.view(chosen_map=self.metrics.get_first_map())
        print("Finished at:")
        self.view()

############# OSCAR WHAT ##############
    def raw_stats(self): 
       return []
########################################

    def get_infected_count(self):
        """
            Determine how many people in total are 
            infected in the current map. 
            Returns:
                int infected_count:     Number of infected individuals 
                                            in the current map. 
        """  
        infected_count = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.map.get_element_at(i, j) == Tile.INFECTED:
                        infected_count +=1
        return infected_count

############# OSCAR PLEASE COMMENT BELOW ##############

    def track_infected_statistics(self):
        infected_stats = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.map.get_element_at(i, j) == Tile.INFECTED:
                    # Count uninfected persons within two spaces
                    uninfected_within_two_spaces = 0
                    squares_to_visit = 0

                    #self.new_list = []

                    for x in range(i-2, i+3):
                        for y in range(j-2, j+3):
                            if x==i and y==j:
                                continue
                            elif self.map.check_in_bounds(x, y):
                                # If the tile is a person and not infected, increase uninfected count
                                if self.map.get_element_at(x, y) == Tile.PERSON:
                                    uninfected_within_two_spaces += 1
                                # Count the number of empty squares the infected person can visit
                                if self.map.get_element_at(x, y) == Tile.EMPTY or self.map.get_element_at(x, y) == Tile.PERSON:
                                    squares_to_visit += 1

                    # Store the statistics for the infected person at (i, j)
                    infected_stats.append(
                        [i,
                        j,
                        uninfected_within_two_spaces,
                        squares_to_visit,
                        self.metrics.iterations
                    ])
                    continue
                continue

        return infected_stats

    def track_overall_statistics(self):
        total_infected = 0
        total_deaths = len(self.metrics.get_dead())
        total_healed = self.metrics.get_healed()

        # Count current number of infected people
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.map.get_element_at(i, j) == Tile.INFECTED:
                    total_infected += 1
        alive_stats = self.metrics.get_avg_alive_stats()
        alive_health = alive_stats[0]
        alive_res = alive_stats[1]
        dead_stats = self.metrics.get_avg_dead_stats()
        dead_health = dead_stats[0]
        dead_res = dead_stats[1]
        # Store the overall statistics at the current time step
        overall_stats = [
            self.metrics.iterations,
            total_infected,
            total_deaths,
            total_healed,
            alive_health,
            alive_res,
            dead_health,
            dead_res
        ]

        return overall_stats

class StepMetrics(): 
    def __init__(self,sim_metrics,map):
        self.metrics : SimMetrics = sim_metrics
        self.metrics.set_first_map(map)

class SimMetrics:
    def __init__(self): 
        self.dead = list()
        self.num_healed = 0
        self.iterations = 0
        self.start_count = 0
        self.initial_hotspots = list()
        self.starting_map : Map = None

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
        return ["hotspots",total_matched, total_possible]

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


    def get_avg_alive_stats(self): 
        return self.starting_map.get_average_alive_stats()
    
    def get_avg_dead_stats(self):
        healths = []
        res = [] 
        for d in self.get_dead():
            cur_person : Person = d
            healths.append(cur_person.starting_health)
            res.append(cur_person.natural_resistance)
        return np.mean(healths), np.mean(res)