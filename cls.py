import random
import pygame
import time
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
        self.people = list()
        self.board =  [ [0] * self.size for i in range(self.size)  ]
        self.generate_array()

    def generate_array(self): 
        for i in range(self.size): 
            for j in range(self.size):
                if random.random() < self.prob_person:
                    self.add_person(i,j,[0.2])
                # pass

    def get_random_person(self): 
        return random.choice(self.people)
    
    
    def add_person(self,i,j,cluster_props):
        self.board[i][j] = Person(prob_nurse=self.prob_nurse)

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

    def step(self): 
        pass
    
    def view(self):
        pygame.init()
        screen_size = 750
        size_offset = float(float(screen_size) / float(self.board_size)) * 1.1
        print("Offset ", size_offset)
        screen = pygame.display.set_mode([screen_size, screen_size])
        screen.fill((255, 255, 255))
        for i in range(self.board_size): 
            for j in range(self.board_size):
                x = i * size_offset
                y = j * size_offset
                # print("(",x,y,end=") , ")
                pygame.draw.circle(screen, (100, 100, 100), (x,y), 3)
        # while True:
        pygame.display.flip()
        time.sleep(10)
        # pygame.quit()

    def show_map(self): 
        return self.map.get_map()
