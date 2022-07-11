import torch
from isaacgym import gymtorch, gymapi


def load_room_from_config(room_dict):
    '''
    Loads room config files into useful list for gym

    Args:
        room_dict: dictionary loaded from yaml config

    Output: 
        wall_coords: list with shape (2 x 2 x n), representing coordinates of n walls
        summit_goal_pos: list, representing coordinate of goal for the summit robot
        box_goal_pos: list, representing coordinate of target position for the robot
    '''

    wall_cfg, summit_goal_cfg = room_dict[
        'wall_cfg'], room_dict['summit_goal_cfg']
    box_goal_cfg = room_dict['box_goal_cfg']

    summit_goal_x = summit_goal_cfg['goal_x']
    summit_goal_y = summit_goal_cfg['goal_y']
    summit_goal_z = summit_goal_cfg['goal_z'] if 'goal_z' in summit_goal_cfg else 0.

    wall_coords = map_to_coord(wall_cfg['walls'])
    summit_goal_pos = [summit_goal_x, summit_goal_y, summit_goal_z]

    if room_dict['use_box_goal']:
        box_goal_x = box_goal_cfg['goal_x']
        box_goal_y = box_goal_cfg['goal_y']
        box_goal_z = box_goal_cfg['goal_z'] if 'goal_z' in box_goal_cfg else 0.
        box_goal_pos = [box_goal_x, box_goal_y, box_goal_z]
    else:
        box_goal_pos = summit_goal_pos

    return wall_coords, summit_goal_pos, box_goal_pos


def map_to_coord(walls):
    '''
    Takes in the map list and returns a list of vectors representing boundaries

    Args: 
        walls: array of n wall objects

    Returns: 
        list with shape (2 x 2 x n) 
    '''

    out = []
    for wall in walls:
        mid_coord = [wall['pos_x'], wall['pos_y']]
        width = wall['length'] / 2
        if wall["orientation"] == 'horizontal':
            x1, x2 = mid_coord[0] - width, mid_coord[0] + width
            y1, y2 = mid_coord[1], mid_coord[1]
        elif wall["orientation"] == 'vertical':
            x1, x2 = mid_coord[0], mid_coord[0]
            y1, y2 = mid_coord[1] - width, mid_coord[1] + width
        out.append([[x1, y1], [x2, y2]])

    return out


def get_wall_bounds(radius, wall_coords):
    '''
    Computes boundary around walls where objects cannot spawn

    Args: 
        radius: float,
        wall_coords: tensor in the form of [[[x1, y1], [x2, y2]], ...]

    Returns: 
        bounds: list in the form of [[[x_min, x_max], [y_min, y_max]], ...]
    '''

    bounds = []
    for coord in wall_coords:
        [[x1, y1], [x2, y2]] = coord
        # note that we can only do this because the walls are either vertical or horizontal
        x_min, x_max = x1 - radius, x2 + radius
        y_min, y_max = y1 - radius, y2 + radius
        # is_vertical = x1 == x2
        # if is_vertical:
        #     x_min, x_max = x1 - radius, x2 + radius
        #     y_min, y_max = y1 - radius, y2 + radius
        # else:
        #     x_min, x_max = x1, x1
        #     y_min, y_max = y1 - radius, y1 + radius
        bounds.append([[x_min, x_max], [y_min, y_max]])

    return bounds


def is_valid_pos(pos, bounds, other_object_pos, r):
    for bound in bounds:
        if collides(pos, bound):
            return False
    for pos2 in other_object_pos:
        if dist(pos, pos2) <= r:
            return False
    return True


def collides(point, bound):
    '''
    Takes a 2D coordinate and bound and returns if the coordinate is within the bound

    Args: 
        point: [x, y],
        bound: [[x1, y1], [x2, y2]]

    Returns: 
        collides: boolean
    '''
    [[x1, x2], [y1, y2]] = bound
    [x, y] = point
    return x1 < x < x2 and y1 < y < y2


def dist(p1, p2):
    [x1, y1] = p1
    [x2, y2] = p2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5
