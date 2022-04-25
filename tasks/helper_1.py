import torch
from isaacgym import gymtorch, gymapi

#


def load_room_from_config(room_dict):
    '''
    Loads room config files into useful list for gym

    Args:
        room_dict: dictionary loaded from yaml config

    Output: 
        wall_coords: list with shape (2 x 2 x n), representing coordinates of n walls
        goal_pos: list, representing coordinate of goal
        num_boxes: int, representing number of boxes

    '''

    wall_cfg, goal_cfg, box_cfg = room_dict['wall_cfg'], room_dict['goal_cfg'], room_dict['box_cfg']

    walls = wall_cfg['walls']

    goal_x = goal_cfg['goal_x']
    goal_y = goal_cfg['goal_y']
    goal_z = goal_cfg['goal_z'] if 'goal_z' in room_dict['goal'] else 0.
    goal_radius = goal_cfg['goal_radius']

    boxes = box_cfg['boxes']
    num_boxes = len(boxes)

    wall_coords = map_to_coord(walls)
    goal_pos = [goal_x, goal_y, goal_z]

    return wall_coords, goal_pos, num_boxes


def map_to_coord(walls):
    '''
    Takes in the map dictionary and returns a list of vectors representing boundaries

    Args: 
        walls: dictionary of all the walls indexed by names

    Returns: 
        list with shape (2 x 2 x n)
    '''

    out = []
    for (_, wall_obj) in walls.items():
        wall = wall_obj
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
