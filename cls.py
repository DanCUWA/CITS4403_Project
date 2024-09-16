import random
PROB_NURSE = 0.2
NUM_CLUSTERS = 3
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
        
    
class Disease:
    def __init__(self): 
        self.infectivity = random.randint(0,0.5)
        self.mortality = random.randint(0,0.5)
    def infect_person(self,map): 
        map.get_random_person().add_disease(self)

class Map:
    def __init__(self,size,k): 
        self.size = size
        self.num_clusters = k
        self.people = list()
        self.board = self.generate_array()

    def generate_array(self): 
        board = [self.size][self.size]
        for i in range(self.size): 
            for j in range(self.size):
                pass

    def add_person(self,i,j,cluster_props):
        self.board[i][j] = Person(cluster_props)

    def get_random_person(self): 
        return random.choice(self.people)

            
