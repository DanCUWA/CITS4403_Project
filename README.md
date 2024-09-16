# CITS4403_Project

## Disease Modeelling/epidimiology in different neighbourhoods

### Description:
- k clusters
  - each cluster has it's own p probability
- p probablity of a neihbourhood square being a person
- x_i people in the i_th neighbourhood initially
  - N nurses are assigned randomly person squares based of population in the neighbourhood
    - e.g. 1 nurse to 100 people
    - N = X_i - #of civilians = X_i - X_i/(1-(Nurse density))
- u_i spaces in the i_th neighbourhood
- D = Population Density
  - Defined per cluster as x_i/u_i
- L is the latency period of the disease
  - still infectious but not being affected (losing health)
- ST is probability of successful treatment
  - value should be high but not 100%

#### Other Rules/Specifications
- if a Nurse in one neighbourhood is infected or used then an x_i resident could relocate after a specific timestep to another neighbourhood to get treatment from a Nurse there
- 


#### Game Cluster Allocation and design
- user specificies the 'k' number of clusters to be used
- from there the display is split into k even by number of squares/pixels but random (connected) segments
- iterate through each cluster and change each element to be covered by a person given p_i probability of assignment in the ith cluster - giving X_i total people in neighbourhood i with 
- 




