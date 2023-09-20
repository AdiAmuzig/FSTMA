# Buffered Uncertainty-Aware Voronoi Cells for Probabilistic Multi-Robot Collision and Deadlock Avoidance

The challenge we’re attempting to solve is how to effectively plan paths and movements in a multi-robot system while accounting for the robots’ faults in movement to minimize collisions and deadlock Via using eandom steps.

## Installation

To set up the environment with __Miniconda\Anaconda__, run the `environment.yml` file with the following command from the command-line/terminal from within the project directory (and anaconda/miniconda installed)

```
conda env create -f environment.yml
```

## Running experiments

There are 3 code files in the project, all cotained within the src folder:
1. kalmanFilter.py - containes the functiones responsible for the progration of the planned and executed pathes according to the dinamics of the agents.
2. voronoiCells.py - containing the functions that create and buff the voronoi cells according to the agents' locations and the uncertianty inlocations.
3. fstma.py - all functions that are directly related to the specific project including initialization of the filters and the different actions that can be taken.

To run the code, first add the following parameters:
    N - size of the map. choosen to be 100 for the experiments in the notebook.
    speed - movement speed of the agents
    safty_radius - minimal required distance between agents
    num_of_agents - number of agents on the map (3-8)
    max_dis - maximal allowed distance between the agents and the goal locations at the end of their runs
    
as a general rule, the following ratio should be followed:
    speed < safty_radius
    speed < max_dis
    
To run the experiments you will need to follow the folloing steps:
1. find the set ofactions according to the given parameters. 
This will give you the actions and abillity to plot the planned results via plt.show().

    min_distances_plan, KFilters, actions, added_time_tot = genoratePlannedPath(
            init_locations[0: num_of_agents], goal_locations[0: num_of_agents],
            num_of_agents, max_dis, speed, colors[0: num_of_agents], N, safty_radius)
2. Following this you can extract the executed path.
this will give you the abillity to plot the executed results via plt.show().

        Locations_x = []
        Locations_y = []
        min_distances_exe = []

        for i in range(num_of_agents):
            X = KFilters[i].runSim(actions[i])
            KFilters[i].plotTrajectory(X, colors[i])
            x_location, y_location = fromMat2LocationVal(X, len(X))
            Locations_x.append(x_location)
            Locations_y.append(y_location)
    
3. To examin the minimal distence between 2 agents along the planned and executed pathes. add he following code:

        plotMinDistances(min_distances_plan, safty_radius)
        plt.title('Minimal distance between agents - Planning')
        plt.show()

        for i in range(len(Locations_x[0])):
            curr_locations = []
            for j in range(num_of_agents):
                curr_locations.append([Locations_x[j][i], Locations_y[j][i]])
            min_distances_exe.append(minDistanceBetweenAllAgents(curr_locations))

        plotMinDistances(min_distances_exe, safty_radius)
        plt.title('Minimal distance between agents - Execution')
        plt.show()

