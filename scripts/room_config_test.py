# this script tests writing & reading yaml config of walls which forms the room for the NAMO agent

import yaml

import os

# example 1
room_walls = {
    'left_bound': {
        'pos_x': -10,
        'pos_y': 0,
        'length': 10,
        'orientation': 'vertical',
    },
    'right_bound': {
        'pos_x': 10,
        'pos_y': 0,
        'length': 10,
        'orientation': 'vertical',
    },
    'top_bound': {
        'pos_x': 0,
        'pos_y': 5,
        'length': 20,
        'orientation': 'horizontal',
    },
    'bottom_bound': {
        'pos_x': 0,
        'pos_y': -5,
        'length': 20,
        'orientation': 'horizontal',
    },
    'wall_1': {
        'pos_x': -5,
        'pos_y': 2.5,
        'length': 5,
        'orientation': 'vertical',
    },
    'wall_2': {
        'pos_x': 0,
        'pos_y': -1.5,
        'length': 7,
        'orientation': 'vertical',
    },
    'wall_3': {
        'pos_x': 1.5,
        'pos_y': 2,
        'length': 3,
        'orientation': 'horizontal',
    },
    'wall_4': {
        'pos_x': 8.5,
        'pos_y': 2,
        'length': 3,
        'orientation': 'horizontal',
    }}

room_dict = {'room_name': 'room_0',  'height': 2.,
             'thickness': 0.1, "walls": room_walls, }


cfg_root_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "cfg/rooms")

filename = 'room_0.yaml'

with open(f'{cfg_root_dir}/{filename}', 'w') as f:
    yaml.dump(room_dict, f, sort_keys=False)

# reading file
with open(f'{cfg_root_dir}/{filename}', 'r') as f:
    a = yaml.load(f, Loader=yaml.loader.SafeLoader)

walls = a['walls']

print([wall['length'] for (name, wall) in walls.items()])
