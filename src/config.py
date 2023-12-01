# This is a configuration file intended to store the most used variables
import torch
import torchvision.transforms as transforms
from src import VDAO_FRAMES_SHAPE

# PIL.PILLOW_VERSION = PIL.__version__


# Correspondence object index
idx2object = {
    1: 'dark-blue box',
    2: 'shoe',
    3: 'camera box',
    4: 'towel',
    5: 'white jar',
    6: 'pink bottle',
    7: 'brown box',
    8: 'black coat',
    9: 'black backpack'
}

object2idx = {
    'dark-blue box': 1,
    'shoe': 2,
    'camera box': 3,
    'towel': 4,
    'white jar': 5,
    'pink bottle': 6,
    'brown box': 7,
    'black coat': 8,
    'black backpack': 9
}

fold2videos = {
    'training': {
        '1': [1, 2, 10, 36, 37, 57, 60],
        '2': [3, 8, 9, 33, 34, 35],
        '3': [4, 11, 12, 38, 39, 40],
        '4': [5, 22, 23, 47, 48, 49],
        '5': [6, 13, 14, 15, 41],
        '6': [7, 19, 20, 21, 45, 46],
        '7': [16, 17, 18, 42, 43, 44],
        '8': [24, 25, 26, 50, 51, 52],
        '9': [27, 28, 29, 30, 31, 32, 53, 54, 55, 56]
    },
    'test': {
        '1': [1, 2, 10, 37],
        '2': [3, 8, 9, 33, 34, 35, 36],
        '3': [4, 11, 12, 38, 39, 40],
        '4': [5, 22, 23, 47, 48, 49],
        '5': [6, 13, 14, 15, 41, 58],
        '6': [7, 19, 20, 21, 45, 46, 59],
        '7': [16, 17, 18, 42, 43, 44],
        '8': [24, 25, 26, 50, 51, 52],
        '9': [27, 28, 29, 30, 31, 32, 53, 54, 55, 56, 57, 58, 59]
    }
}

video2fold = {
    '1': 1,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
    '6': 5,
    '7': 6,
    '8': 2,
    '9': 2,
    '10': 1,
    '11': 3,
    '12': 3,
    '13': 5,
    '14': 5,
    '15': 5,
    '16': 7,
    '17': 7,
    '18': 7,
    '19': 6,
    '20': 6,
    '21': 6,
    '22': 4,
    '23': 4,
    '24': 8,
    '25': 8,
    '26': 8,
    '27': 9,
    '28': 9,
    '29': 9,
    '30': 9,
    '31': 9,
    '32': 9,
    '33': 2,
    '34': 2,
    '35': 2,
    '36': 1,
    '37': 1,
    '38': 3,
    '39': 3,
    '40': 3,
    '41': 5,
    '42': 7,
    '43': 7,
    '44': 7,
    '45': 6,
    '46': 6,
    '47': 4,
    '48': 4,
    '49': 4,
    '50': 8,
    '51': 8,
    '52': 8,
    '53': 9,
    '54': 9,
    '55': 9,
    '56': 9,
    '57': 1,
    '58': 5,
    '59': 6,
    '60': 1,
}

idx2tarname = {
    1: 'obj-sing-amb-part01-video06.avi',
    2: 'obj-sing-amb-part01-video04.avi',
    3: 'obj-sing-amb-part01-video02.avi',
    4: 'obj-sing-amb-part01-video09.avi',
    5: 'obj-sing-amb-part03-video06.avi',
    6: 'obj-sing-ext-part02-video02.avi',
    7: 'obj-sing-ext-part03-video03.avi',
    8: 'obj-sing-amb-part01-video01.avi',
    9: 'obj-sing-amb-part01-video03.avi',
    10: 'obj-sing-amb-part01-video07.avi',
    11: 'obj-sing-amb-part01-video08.avi',
    12: 'obj-sing-amb-part01-video10.avi',
    13: 'obj-sing-amb-part02-video01.avi',
    14: 'obj-sing-amb-part02-video02.avi',
    15: 'obj-sing-amb-part02-video03.avi',
    16: 'obj-sing-amb-part02-video04.avi',
    17: 'obj-sing-amb-part02-video05.avi',
    18: 'obj-sing-amb-part03-video01.avi',
    19: 'obj-sing-amb-part03-video02.avi',
    20: 'obj-sing-amb-part03-video04.avi',
    21: 'obj-sing-amb-part03-video05.avi',
    22: 'obj-sing-amb-part03-video07.avi',
    23: 'obj-sing-amb-part03-video08.avi',
    24: 'obj-sing-amb-part03-video09.avi',
    25: 'obj-sing-amb-part03-video10.avi',
    26: 'obj-sing-amb-part03-video11.avi',
    27: 'obj-sing-amb-part03-video12.avi',
    28: 'obj-sing-amb-part03-video13.avi',
    29: 'obj-sing-amb-part03-video14.avi',
    30: 'obj-sing-amb-part03-video15.avi',
    31: 'obj-sing-amb-part03-video16.avi',
    32: 'obj-sing-amb-part03-video17.avi',
    33: 'obj-sing-ext-part01-video01.avi',
    34: 'obj-sing-ext-part01-video02.avi',
    35: 'obj-sing-ext-part01-video03.avi',
    36: 'obj-sing-ext-part01-video04.avi',
    37: 'obj-sing-ext-part01-video06.avi',
    38: 'obj-sing-ext-part01-video07.avi',
    39: 'obj-sing-ext-part01-video08.avi',
    40: 'obj-sing-ext-part01-video09.avi',
    41: 'obj-sing-ext-part02-video03.avi',
    42: 'obj-sing-ext-part02-video04.avi',
    43: 'obj-sing-ext-part02-video05.avi',
    44: 'obj-sing-ext-part03-video01.avi',
    45: 'obj-sing-ext-part03-video02.avi',
    46: 'obj-sing-ext-part03-video04.avi',
    47: 'obj-sing-ext-part03-video05.avi',
    48: 'obj-sing-ext-part03-video06.avi',
    49: 'obj-sing-ext-part03-video07.avi',
    50: 'obj-sing-ext-part03-video08.avi',
    51: 'obj-sing-ext-part03-video09.avi',
    52: 'obj-sing-ext-part03-video10.avi',
    53: 'obj-sing-ext-part03-video11.avi',
    54: 'obj-sing-ext-part03-video12.avi',
    55: 'obj-sing-ext-part03-video13.avi',
    56: 'obj-sing-ext-part03-video14.avi',
    57: 'obj-sing-amb-part01-video05.avi',
    58: 'obj-sing-ext-part02-video01.avi',
    59: 'obj-sing-amb-part03-video03.avi',
    60: 'obj-sing-ext-part01-video05.avi'
}

index2refname = {
    1: 'ref-sing-amb-part01-video01.avi',
    2: 'ref-sing-amb-part03-video01.avi',
    3: 'ref-sing-ext-part02-video01.avi',
    4: 'ref-sing-ext-part03-video01.avi',
    5: 'ref-sing-amb-part02-video01.avi',
    6: 'ref-sing-ext-part01-video01.avi',
    7: 'ref-sing-amb-part01-video02.avi'
}

# setting device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')

# torch transforms
resize_transform_half = transforms.Resize(
    (VDAO_FRAMES_SHAPE[0] // 2, VDAO_FRAMES_SHAPE[1] // 2))
resize_transform_quarter = transforms.Resize(
    (VDAO_FRAMES_SHAPE[0] // 4, VDAO_FRAMES_SHAPE[1] // 4))
to_tensor_transform = transforms.ToTensor()

# Composed transformations
transformations_half = transforms.Compose(
    [resize_transform_half, to_tensor_transform])
transformations_quarter = transforms.Compose(
    [resize_transform_quarter, to_tensor_transform])

# Fold split
fold_split = {
    1: {
        1: {
            'training': [3, 4, 6, 7, 9],
            'validation': [2, 5, 8],
        },
        2: {
            'training': [2, 4, 5, 7, 8],
            'validation': [3, 6, 9],
        },
        3: {
            'training': [3, 5, 6, 8, 9],
            'validation': [2, 4, 7],
        },
        4: {
            'training': [2, 4, 6, 7, 9],
            'validation': [3, 5, 8],
        },
        5: {
            'training': [2, 3, 5, 7, 8],
            'validation': [4, 6, 9],
        },
        6: {
            'training': [3, 4, 6, 8, 9],
            'validation': [2, 5, 7],
        },
        7: {
            'training': [2, 4, 5, 7, 9],
            'validation': [3, 6, 8],
        },
        8: {
            'training': [2, 3, 5, 6, 8],
            'validation': [4, 7, 9],
        }
    },
    2: {
        1: {
            'training': [3, 4, 6, 7, 9],
            'validation': [1, 5, 8],
        },
        2: {
            'training': [1, 4, 5, 7, 8],
            'validation': [3, 6, 9],
        },
        3: {
            'training': [3, 5, 6, 8, 9],
            'validation': [1, 4, 7],
        },
        4: {
            'training': [1, 4, 6, 7, 9],
            'validation': [3, 5, 8],
        },
        5: {
            'training': [1, 3, 5, 7, 8],
            'validation': [4, 6, 9],
        },
        6: {
            'training': [3, 4, 6, 8, 9],
            'validation': [1, 5, 7],
        },
        7: {
            'training': [1, 4, 5, 7, 9],
            'validation': [3, 6, 8],
        },
        8: {
            'training': [1, 3, 5, 6, 8],
            'validation': [4, 7, 9],
        }
    },
    3: {
        1: {
            'training': [2, 4, 6, 7, 9],
            'validation': [1, 5, 8],
        },
        2: {
            'training': [1, 4, 5, 7, 8],
            'validation': [2, 6, 9],
        },
        3: {
            'training': [2, 5, 6, 8, 9],
            'validation': [1, 4, 7],
        },
        4: {
            'training': [1, 4, 6, 7, 9],
            'validation': [2, 5, 8],
        },
        5: {
            'training': [1, 2, 5, 7, 8],
            'validation': [4, 6, 9],
        },
        6: {
            'training': [2, 4, 6, 8, 9],
            'validation': [1, 5, 7],
        },
        7: {
            'training': [1, 4, 5, 7, 9],
            'validation': [2, 6, 8],
        },
        8: {
            'training': [1, 2, 5, 6, 8],
            'validation': [4, 7, 9],
        }
    },
    4: {
        1: {
            'training': [2, 3, 6, 7, 9],
            'validation': [1, 5, 8],
        },
        2: {
            'training': [1, 3, 5, 7, 8],
            'validation': [2, 6, 9],
        },
        3: {
            'training': [2, 5, 6, 8, 9],
            'validation': [1, 3, 7],
        },
        4: {
            'training': [1, 3, 6, 7, 9],
            'validation': [2, 5, 8],
        },
        5: {
            'training': [1, 2, 5, 7, 8],
            'validation': [3, 6, 9],
        },
        6: {
            'training': [2, 3, 6, 8, 9],
            'validation': [1, 5, 7],
        },
        7: {
            'training': [1, 3, 5, 7, 9],
            'validation': [2, 6, 8],
        },
        8: {
            'training': [1, 2, 5, 6, 8],
            'validation': [3, 7, 9],
        }
    },
    5: {
        1: {
            'training': [2, 3, 6, 7, 9],
            'validation': [1, 4, 8],
        },
        2: {
            'training': [1, 3, 4, 7, 8],
            'validation': [2, 6, 9],
        },
        3: {
            'training': [2, 4, 6, 8, 9],
            'validation': [1, 3, 7],
        },
        4: {
            'training': [1, 3, 6, 7, 9],
            'validation': [2, 4, 8],
        },
        5: {
            'training': [1, 2, 4, 7, 8],
            'validation': [3, 6, 9],
        },
        6: {
            'training': [2, 3, 6, 8, 9],
            'validation': [1, 4, 7],
        },
        7: {
            'training': [1, 3, 4, 7, 9],
            'validation': [2, 6, 8],
        },
        8: {
            'training': [1, 2, 4, 6, 8],
            'validation': [3, 7, 9],
        }
    },
    6: {
        1: {
            'training': [2, 3, 5, 7, 9],
            'validation': [1, 4, 8],
        },
        2: {
            'training': [1, 3, 4, 7, 8],
            'validation': [2, 5, 9],
        },
        3: {
            'training': [2, 4, 5, 8, 9],
            'validation': [1, 3, 7],
        },
        4: {
            'training': [1, 3, 5, 7, 9],
            'validation': [2, 4, 8],
        },
        5: {
            'training': [1, 2, 4, 7, 8],
            'validation': [3, 5, 9],
        },
        6: {
            'training': [2, 3, 5, 8, 9],
            'validation': [1, 4, 7],
        },
        7: {
            'training': [1, 3, 4, 7, 9],
            'validation': [2, 5, 8],
        },
        8: {
            'training': [1, 2, 4, 5, 8],
            'validation': [3, 7, 9],
        }
    },
    7: {
        1: {
            'training': [2, 3, 5, 6, 9],
            'validation': [1, 4, 8],
        },
        2: {
            'training': [1, 3, 4, 6, 8],
            'validation': [2, 5, 9],
        },
        3: {
            'training': [2, 4, 5, 8, 9],
            'validation': [1, 3, 6],
        },
        4: {
            'training': [1, 3, 5, 6, 9],
            'validation': [2, 4, 8],
        },
        5: {
            'training': [1, 2, 4, 6, 8],
            'validation': [3, 5, 9],
        },
        6: {
            'training': [2, 3, 5, 8, 9],
            'validation': [1, 4, 6],
        },
        7: {
            'training': [1, 3, 4, 6, 9],
            'validation': [2, 5, 8],
        },
        8: {
            'training': [1, 2, 4, 5, 8],
            'validation': [3, 6, 9],
        }
    },
    8: {
        1: {
            'training': [2, 3, 5, 6, 9],
            'validation': [1, 4, 7],
        },
        2: {
            'training': [1, 3, 4, 6, 7],
            'validation': [2, 5, 9],
        },
        3: {
            'training': [2, 4, 5, 7, 9],
            'validation': [1, 3, 6],
        },
        4: {
            'training': [1, 3, 5, 6, 9],
            'validation': [2, 4, 7],
        },
        5: {
            'training': [1, 2, 4, 6, 7],
            'validation': [3, 5, 9],
        },
        6: {
            'training': [2, 3, 5, 7, 9],
            'validation': [1, 4, 6],
        },
        7: {
            'training': [1, 3, 4, 6, 9],
            'validation': [2, 5, 7],
        },
        8: {
            'training': [1, 2, 4, 5, 7],
            'validation': [3, 6, 9],
        }
    },
    9: {
        1: {
            'training': [2, 3, 5, 6, 8],
            'validation': [1, 4, 7],
        },
        2: {
            'training': [1, 3, 4, 6, 7],
            'validation': [2, 5, 8],
        },
        3: {
            'training': [2, 4, 5, 7, 8],
            'validation': [1, 3, 6],
        },
        4: {
            'training': [1, 3, 5, 6, 8],
            'validation': [2, 4, 7],
        },
        5: {
            'training': [1, 2, 4, 6, 7],
            'validation': [3, 5, 8],
        },
        6: {
            'training': [2, 3, 5, 7, 8],
            'validation': [1, 4, 6],
        },
        7: {
            'training': [1, 3, 4, 6, 8],
            'validation': [2, 5, 7],
        },
        8: {
            'training': [1, 2, 4, 5, 7],
            'validation': [3, 6, 8],
        }
    }
}
