from typing import List
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import math


def voroniCellsLineStrings(points: np.array) -> List[LineString]:
    """Generate the lines that create the voronoi cells from a list of point (2D)

    Args:
        points (np.array): List of points to use in deviding the space into voronoi cells

    Returns:
        List[LineString]: List of lines that make up the voronoi cells
    """
    vor = Voronoi(points)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    voronoi_lines = []

    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            voronoi_lines.append(LineString(vor.vertices[simplex]))
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            voronoi_lines.append(LineString([vor.vertices[i], far_point]))

    return voronoi_lines


def plotVoronoiCellsLines(voronoi_lines: List[LineString]) -> None:
    """Plots (but does not display) the voronoi cells in the form of lines

    Args:
        voronoi_lines (List[LineString]): List of lines that make up the voronoi cells
    """
    for i in range(len(voronoi_lines)):
        x, y = voronoi_lines[i].xy
        plt.plot(x, y, 'k')


def bufferLineInPointDirrection(line: LineString, added_size: float, point: List[float]) -> Polygon:
    """Enlarges a line by a given size toword a point in space

    Args:
        line (LineString): line to enlarge
        added_size (float): size by which to enlarge
        dir (List[float], optional): direction by which to enlargethe line 

    Returns:
        Polygon: The new line after buffing
    """
    epsilon = 0.001

    # calcualting the new points to add to the polygon
    x, y = line.xy

    # cosin sentence
    len_line = math.dist([x[0], y[0]], [x[1], y[1]])
    len_p2line1 = math.dist([x[0], y[0]], point)
    len_p2line2 = math.dist([x[1], y[1]], point)
    acos_theta = (len_p2line1**2 - len_line**2 - len_p2line2**2) / \
        (-2 * len_line * len_p2line2)

    # vector values to enlarge the line in the right direction
    d = len_p2line2 * acos_theta
    line_vector = [(x[0] - x[1])/len_line,
                   (y[0] - y[1])/len_line]
    new_point = [x[1] + d * line_vector[0],
                 y[1] + d * line_vector[1]]
    len_p2p = math.dist(new_point, point)
    if len_p2p > epsilon:
        points_vector = [(point[0] - new_point[0])/len_p2p,
                         (point[1] - new_point[1])/len_p2p]

        x_add = added_size * points_vector[0]
        y_add = added_size * points_vector[1]

    else:
        x_add = epsilon
        y_add = epsilon

    point1 = (x[0], y[0])
    point2 = (x[1], y[1])
    point3 = (x[1] + x_add, y[1] + y_add)
    point4 = (x[0] + x_add, y[0] + y_add)

    line_buffed = Polygon([point1, point2, point3, point4])
    return line_buffed


def minDistanceFromPoint2Lines(line: LineString, point: List[float], points: np.array) -> bool:
    """Check wether the given line is closest to a given point in comparison to a list of points (2D)

    Args:
        line (LineString): Line to examin 
        arr_point (List[float]): Point to check if closest to
        arr_points (np.array): A list of points to compare

    Returns:
        bool: Is the point closest to the line then all other points
    """
    dis_min = Point(point).distance(line)
    for p in points:
        new_dis = Point(p).distance(line)
        if new_dis < dis_min:
            return False
    return True


def genorateVoroniEdgesToPoints(points: np.array, voroni_lines: List[LineString]) -> List[List[LineString]]:
    """Create a list containing lists of lines in the voronoi cells 
    connected to the appropriat points in space, 
    the ones that are closest to the lines

    Args:
        points (np.array):List of 2D points
        voroni_lines (List[LineString]): List of lines that make up the voronoi cells

    Returns:
        List[List[LineString]]: List of lists of the lines composing the voronoi cells 
        acording to the relevant point
    """
    lines_of_points = []

    for point in points:
        lines = []

        for line in voroni_lines:

            if minDistanceFromPoint2Lines(line, point, points):
                lines.append(line)

        lines_of_points.append(lines)

    return lines_of_points


def bufferOneVoronoiCell(points: np.array, point_num: int, voroni_lines: List[LineString], added_size: float) -> List[Polygon]:
    """Buffer an existing voronoi cell toward a point according to a given buffer

    Args:
        points (np.array): List of point that are responsible for the voronoi cells
        point_num (int): Specific point to buffer
        voroni_lines (List[LineString]): The voronoi lines that represent the voronoi cells
        added_size (float): The size by which to buff the voronoi cell towards the requested point

    Returns:
        List[Polygon]: List containing the new poligons that represent the buffed voronoi cell
        according to the demanded buffer toward the specified point
    """
    lines_of_points = genorateVoroniEdgesToPoints(points, voroni_lines)
    lines_of_point = lines_of_points[point_num]
    point = points[point_num]
    buffed_cell = []

    for line in lines_of_point:
        buffed_line = bufferLineInPointDirrection(line, added_size, point)
        buffed_cell.append(buffed_line)

    return buffed_cell


def bufferAllVoronoiCell(points: np.array, voroni_lines: List[LineString], added_sizes: List[float]) -> List[Polygon]:
    """Buffer The existing voronoi cells toward each point according to individial given buffers

    Args:
        points (np.array): List of point that are responsible for the voronoi cells
        voroni_lines (List[LineString]): The voronoi lines that represent the voronoi cells
        added_sizes (List[float]): The sizes by which to buff each voronoi cell towards its respected point

    Returns:
        List[Polygon]: List containing the new poligons that represent the buffed voronoi cells
        according to the demanded buffer
    """
    buffed_lines = []
    for point_num in range(len(points)):
        buffed_cell = bufferOneVoronoiCell(
            points, point_num, voroni_lines, added_sizes[point_num])
        buffed_lines += buffed_cell
    return buffed_lines


def plotBuffedVoronoiCells(buffed_cells: List[Polygon]) -> None:
    """Plot all voronoi cells

    Args:
        buffed_cells (List[Polygon]): Buffed voronoi cells
    """
    for line in buffed_cells:
        x, y = line.exterior.xy
        plt.fill(x, y, 'k')


def checkPointsInBuffedLines(buffed_cells: List[Polygon], location: List[float]) -> bool:
    """Check wether or not a given location is within the buffed voronoi lines

    Args:
        buffed_cells (List[Polygon]): Buffed voronoi cells
        location (List[bool]): Location in a 2D map

    Returns:
        bool: Wether or not a given location is within the buffed voronoi lines
    """
    point = Point(location[0], location[1])
    for line in buffed_cells:
        if line.intersects(point):
            return True
    return False
