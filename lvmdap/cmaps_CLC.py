import numpy as np
from matplotlib import use as mpl_use
mpl_use('Agg')

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors



def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))



def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


#
#
# This is a great colormap to represent velocity fields
#
reds = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6"]
blues = ["#531132", "#800d20", "#aa2c24", "#db6d52", "#f1b691"]
#black = ["#423f46"]
black = ["#ece75f"]

#
# We order the colors!
#
hex_list_CLC = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6","#423f46","#f1b691", "#db6d52", "#aa2c24", "#800d20", "#531132"]
#hex_list = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6","#ece75f","#f1b691", "#db6d52", "#aa2c24", "#800d20", "#531132"]
hex_list = ["#01025a", "#193476", "#376b94", "#5ea1ba", "#a3dbe6","#f1ee8e","#f1b691", "#db6d52", "#aa2c24", "#800d20", "#531132"]



def vel_map():
	return get_continuous_cmap(hex_list)


def vel_map_CLC():
	return get_continuous_cmap(hex_list_CLC)




