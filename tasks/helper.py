import torch


def map_to_coord(dict):
    '''
    Takes in the map dictionary and returns a list of vectors representing boundaries

    Input: 
    map dictionary (generated from yaml configs): dict

    Output: 
    tensor representing boundaries: tensor with shape (2 x 2 x n)
    '''

    walls = dict["walls"]

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

    return torch.tensor(out)
