from cls import *
import time
BOARDSIZE = 25
NUM_CLUSTERS = 1
PROB_NURSE = 0.02
PROB_PERSON = 0.2
def main():
    sim = Simulation(board_size=BOARDSIZE,num_clusters=NUM_CLUSTERS,
                     prob_nurse=PROB_NURSE,prob_person=PROB_PERSON)
    # print(sim.show_map())
    sim.start()
    sim.view()
    while True:
        sim.step()
        sim.view()
        time.sleep(3)
if __name__ == "__main__": 
    main()
