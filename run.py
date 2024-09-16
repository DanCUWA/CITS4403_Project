from cls import *
BOARDSIZE = 100
NUM_CLUSTERS = 4
PROB_NURSE = 0.2
PROB_PERSON = 0.5 
def main():
    sim = Simulation(board_size=BOARDSIZE,num_clusters=NUM_CLUSTERS,
                     prob_nurse=PROB_NURSE,prob_person=PROB_PERSON)
    print(sim.show_map())
    sim.view()
if __name__ == "__main__": 
    main()