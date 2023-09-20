from src.kalmanFilter import*
from src.voroniCells import*


def plotRelativeAddedTime(num_of_agents_vec: List[int], added_time_vec: List[float]) -> None:
    """Plot the relative added number of steps to the number of agents

    Args:
        num_of_agents_vec (List[int]): Number of agents that tried to switch places
        added_time_vec (List[float]): Relative number of time steps added to the path in relation to total path length
    """
    plt.bar(num_of_agents_vec, added_time_vec, color='blue')
    plt.title('Relative added time due to random steps')
    plt.xlabel('Number of agents')
    plt.ylabel('Added time steps/length of path')
    plt.grid(color='lightgray', linestyle='--')
    plt.autoscale()


def actionsWithPositions(curr_pos: List[float], next_pos: List[float], speed: float) -> np.matrix:
    """Given an initial and goal location and the movement speed of an agent,
    get the next action to get from the initial to the goal location

    Args:
        curr_pos (List[float]): initial location
        next_pos (List[float]): goal location
        action_speed (float): movement speed of the agent

    Returns:
        np.matrix: Action to perform
    """

    deg = np.arctan2(
        (next_pos[1] - curr_pos[1]),
        (next_pos[0] - curr_pos[0]))
    x_speed = speed * np.cos(deg)
    y_speed = speed * np.sin(deg)
    action = np.matrix([[x_speed], [y_speed]])

    return action


def plotMinDistances(min_distances: List[float], safty_radius: float) -> None:
    """Plot te minimal distance between 2 agetnts

    Args:
        min_distances (List[float]): Minimal distance between 2 agents
        safty_radius (float): Safty radius between every 2 agents
    """
    time_steps = list(range(len(min_distances)))
    safty_vec = [safty_radius] * (len(min_distances))

    plt.plot(time_steps, min_distances, color='blue',
             label="Minimal distance between 2 agents")
    plt.plot(time_steps, safty_vec, 'r--', label="Safty Radius")
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Distance')
    plt.grid(color='lightgray', linestyle='--')
    plt.autoscale()


def genorateKalmanFilters(init_locations: List[List[float]]) -> Tuple[List[KalmanFilter], List[belief]]:
    """Initiate the Kalman filters that will estimate the location uncertainty
    of the agents along the planned path.

    Args:
        init_locations (List[List[float]]): initial location of all agents

    Returns:
        Tuple[List[KalmanFilter], List[belief]]:
            List[KalmanFilter] - KFilters - The Kalman filters of all agents
            List[belief] - Beliefs - Belief states for all agents, containing only the initial belief stats
    """
    # initiate Kalman filter for the agents
    A = np.matrix('1 0; 0 1')
    B = np.matrix('1 0; 0 1')
    C = np.matrix('1 0; 0 1')
    R = np.matrix('0.01 0; 0 0.01')
    Q = np.matrix('0.01 0; 0 0.01')
    Sigma_0 = np.matrix('0 0 ; 0 0')
    k = 11.82  # value for 3Sigma meaning 99.74% probability

    KFilters = []
    Beliefs = []
    for pos in init_locations:
        x_0 = np.matrix([[pos[0]], [pos[1]]])
        mu_0 = x_0
        AgentKF = KalmanFilter(A, B, C, R, Q, mu_0, Sigma_0, x_0, k)
        Beliefs.append(AgentKF.belief_0)
        KFilters.append(AgentKF)

    return KFilters, Beliefs


def simulateOneAgentOneStep(curr_loc: List[float], goal_loc: List[float], speed: float, KF: KalmanFilter, Belief: belief, safty_radius: float) -> Tuple[float, List[float], belief, np.matrix]:
    """simulate the estimated progress of a single agent one step forward toward the goal

    Args:
        curr_loc (List[float]): Current location
        goal_loc (List[float]): Goal location
        action_speed (float): Action speed of the agent
        KF (KalmanFilter): Kalman filter of the agent
        Belief (belief): Last belief state of the agent
        safty_radius (float): Safty radius arount the agent

    Returns:
        Tuple[float, List[float], belief]:
            float - buffer_size -  size of the buffer for the voronoi line
            List[float] - mean - estimated mean location of the agent
            belief - belief_new - new belief state of the agent
            np.matrix - action - the action taken from the previus location to the next
    """

    action = actionsWithPositions(curr_loc, goal_loc, speed)
    belief_new = KF.PropagateUpdateBelief(Belief, action)
    Belief = belief_new
    new_sigma = belief_new.Sigma.max()
    buffer_size = new_sigma + safty_radius
    mean = np.squeeze(np.asarray(belief_new.mu))

    return buffer_size, mean, belief_new, action


def randomStep(curr_loc: List[float], speed: float, KF: KalmanFilter, Belief: belief, N: float, safty_radius: float) -> Tuple[float, List[float], belief]:
    """simulate the estimated progress of a single agent one step forward toward a random location

    Args:
        curr_loc (List[float]): Current location
        action_speed (float): Action speed of the agent
        KF (KalmanFilter): Kalman filter of the agent
        Belief (belief): Last belief state of the agent
        N (float): Map size N x N
        safty_radius (float): safty radius around the agent

    Returns:
        Tuple[float, List[float], belief]:
            float - buffer_size -  size of the buffer for the voronoi line
            List[float] - estimated mean location of the agent
            belief - belief_new - new belief state of the agent
    """
    goal_x = np.random.uniform(0, N)
    goal_y = np.random.uniform(0, N)
    goal_loc = [goal_x, goal_y]
    return simulateOneAgentOneStep(curr_loc, goal_loc, speed, KF, Belief, safty_radius)


def maxDistanceForAllAgent(curr_locations: List[List[float]], goal_locations: List[List[float]]) -> float:
    """Calculate the maximal distance between the current locations and the goal locations

    Args:
        curr_locations (List[List[float]]): current locations in 2D
        goal_locations (List[List[float]]): Goal locations in 2D

    Returns:
        float: Maximal distance between the currents locations and the goal locations
    """
    max_dis = 0
    for i in range(len(curr_locations)):
        distance = distanceBetweenTwoPoints(
            curr_locations[i], goal_locations[i])
        if distance > max_dis:
            max_dis = distance
    return max_dis


def distanceBetweenTwoPoints(point1: List[float], point2: List[float]) -> float:
    """calculate the distance between two points

    Args:
        point1 (List[float]): point location in 2D
        point1 (List[float]): point location in 2D

    Returns:
        float: Distance between two points
    """
    x_dis = abs(point1[0] - point2[0])
    y_dis = abs(point1[1] - point2[1])
    distance = math.sqrt(x_dis**2 + y_dis**2)
    return distance


def minDistanceBetweenAllAgents(curr_locations: List[List[float]]) -> float:
    """Calculate the minimal distance between all agents

    Args:
        curr_locations (List[List[float]]): current locations in 2D

    Returns:
        float: Minimal distance between all agents
    """
    min_dis = distance = distanceBetweenTwoPoints(
        curr_locations[0], curr_locations[1])

    for i in range(len(curr_locations)):
        for j in range(len(curr_locations)):
            if i != j:
                distance = distanceBetweenTwoPoints(
                    curr_locations[i], curr_locations[j])
                if distance < min_dis:
                    min_dis = distance
    return min_dis


def genoratePlannedPath(init_locations: List[List[float]], goal_locations: List[List[float]], num_of_agents: int, max_dis: float, speed: float, colors: List[Str], N: float, safty_radius: float) -> Tuple[List[float], List[KalmanFilter], List[np.matrix]]:
    """According to the initial locaions and goal locations of the agents,
    genorate a planned path for the agents

    Args:
        init_locations (List[List[float]]): Initial locations of the agents in a 2D map
        goal_locations (List[List[float]]): Goal locations of the agents in a 2D map
        num_of_agents (int): Number of agents on the map
        max_dis (float): Maximal allowed distance from the agents to the goal positions at the end of the run
        speed (float): Movement speed of the agents
        colors (List[Str]): Colord by which to plot the agents
        N (float): Map size N x N
        safty_radius (float): Safty radius around the agents

    Returns:
        Tuple[List[float], List[KalmanFilter], List[np.matrix]]:
            Tuple[List[float] - min_distances - Minimal distances between 2 agents along the path
            List[KalmanFilter] - KFilters - The Kalman filters of all agents
            List[np.matrix] - actions - the actions taken from the initial location up to the goal location

    """
    min_distances = []
    path_planned = []
    actions = [[] for i in range(0, num_of_agents)]

    path_planned.append(init_locations)
    vor = voroniCellsLineStrings(init_locations)
    buffer_vec = np.full(num_of_agents, safty_radius)
    vor_buff = bufferAllVoronoiCell(init_locations, vor, buffer_vec)

    KFilters, Beliefs = genorateKalmanFilters(init_locations)

    plt.title("Planned movements by the agents")
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.grid(color='lightgray', linestyle='--')
    plt.autoscale()

    added_time_tot = 0

    while maxDistanceForAllAgent(path_planned[-1], goal_locations) > max_dis:
        mean_locations = []
        buffer_vec = []

        added_time = 0

        for j in range(num_of_agents):
            buffer_size, mean, belief_new, action = simulateOneAgentOneStep(
                path_planned[-1][j], goal_locations[j],
                speed, KFilters[j], Beliefs[j], safty_radius)

            loops = 0
            while checkPointsInBuffedLines(vor_buff, mean) == True:
                added_time = 1
                if loops < 100:
                    buffer_size, mean, belief_new, action = randomStep(
                        path_planned[-1][j], speed, KFilters[j],
                        Beliefs[j], N, safty_radius)
                    loops += 1
                else:
                    mean = np.squeeze(np.asarray(Beliefs[j].mu))
                    belief_new = belief(Beliefs[j].mu, belief_new.Sigma)
                    action = np.matrix('0 ; 0')
                    break

                # if loops < 100:
                #     new_speed = speed * (1 - loops / 100)
                #     buffer_size, mean, belief_new, action = simulateOneAgentOneStep(
                #         path_planned[-1][j], goal_locations[j],
                #         new_speed, KFilters[j], Beliefs[j])
                #     loops += 1
                # else:
                #     mean = np.squeeze(np.asarray(Beliefs[j].mu))
                #     belief_new = belief(Beliefs[j].mu, belief_new.Sigma)
                #     action = np.matrix('0 ; 0')
                #     break

            actions[j].append(action)
            Beliefs[j] = belief_new
            buffer_vec.append(buffer_size)
            mean_locations.append([mean[0], mean[1]])

            plt.plot(mean[0], mean[1], color=colors[j], marker='o')

        mean_locations = np.array(mean_locations)
        path_planned.append(mean_locations)
        vor = voroniCellsLineStrings(mean_locations)
        vor_buff = bufferAllVoronoiCell(mean_locations, vor, buffer_vec)
        min_distances.append(minDistanceBetweenAllAgents(mean_locations))

        added_time_tot = added_time_tot + added_time
        # plotBuffedVoronoiCells(vor_buff)

        # plt.pause(0.1)
        # plt.show()
        # plt.clf()

    return min_distances, KFilters, actions, added_time_tot


if __name__ == "__main__":
    N = 100
    speed = 2
    safty_radius = 3
    num_of_agents = 4
    max_dis = 5
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'silver']

    init_locations = [[1, 1], [N - 1, N - 1], [1, N - 1], [N - 1, 1],
                      [N/2, 1], [N/2, N - 1], [1, N/2], [N - 1, N/2]]
    goal_locations = [[N - 1, N - 1], [1, 1], [N - 1, 1], [1, N - 1],
                      [N/2, N - 1], [N/2, 1], [N - 1, N/2], [1, N/2]]

    # added_time_vec = []

    num_of_agents_vec = [4, 6, 8]
    added_time_vec = [0.29, 0.42, 0.51]
    plotRelativeAddedTime(num_of_agents_vec, added_time_vec)
    plt.show()

    for num_of_agents in num_of_agents_vec:
        print(num_of_agents)
        relative_added_time = []
        for i in range(10):
            min_distances_plan, KFilters, actions, added_time = genoratePlannedPath(
                init_locations[0: num_of_agents], goal_locations[0: num_of_agents],
                num_of_agents, max_dis, speed, colors[0: num_of_agents], N, safty_radius)
            relative_added_time.append(added_time/len(actions[0]))
            print(i)

        added_time_vec.append(sum(relative_added_time) /
                              len(relative_added_time))

    # plt.show()
    plotRelativeAddedTime(num_of_agents_vec, added_time_vec)
    plt.show()

    plotMinDistances(min_distances_plan, safty_radius)
    plt.title('Minimal distance between agents - Planning')
    plt.show()

    Locations_x = []
    Locations_y = []
    min_distances_exe = []

    for i in range(num_of_agents):
        X = KFilters[i].runSim(actions[i])
        KFilters[i].plotTrajectory(X, colors[i])
        x_location, y_location = fromMat2LocationVal(X, len(X))
        Locations_x.append(x_location)
        Locations_y.append(y_location)

    plt.show()

    for i in range(len(Locations_x[0])):
        curr_locations = []
        for j in range(num_of_agents):
            curr_locations.append([Locations_x[j][i], Locations_y[j][i]])
        min_distances_exe.append(minDistanceBetweenAllAgents(curr_locations))

    plotMinDistances(min_distances_exe, safty_radius)
    plt.title('Minimal distance between agents - Execution')
    plt.show()
