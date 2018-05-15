import numpy as np

def move_breakpoints(breakpoints, mode='match'):
    '''
    Update the configuration of the breakpoints:
    1) check if there are more breakpoints or more empty sites
    2) select the smallest group
    3) generate_neighbour() configuration for as many times as there are members in the selected group
    4) convert back to breakpoints if we are moving the voids.
    '''
    # 0) specific to match/mismatch breakpoints:
    if mode == 'match':
        max_complexity = 18
    elif mode == 'mismatch':
        max_complexity = 19

    print "entered module"

    # 1) Do we move the breakpoints or the empty sites?
    nmbr_breakpoints = len(breakpoints)
    nmbr_empty = max_complexity - nmbr_breakpoints



    # A)If majority is empty, move the breakpoints:
    if nmbr_breakpoints <= nmbr_empty:
        # A1) Move the breakpoints:
        for i in range(nmbr_breakpoints):
            breakpoints = generate_neighbour(breakpoints, max_complexity)

    # B)If majority is breakpoints, move the empty sites:
    else:
        # B1) Determine the location of the sites without a breakpoint:
        empty_sites = convert_breakpoints_empty(breakpoints, max_complexity)

        # B2) Move the empty sites:
        for i in range(nmbr_empty):
            empty_sites = generate_neighbour(empty_sites, max_complexity)

        # B3) Convert back to list of breakpoint locations:
        new_breakpoints = convert_breakpoints_empty(empty_sites, max_complexity)
        breakpoints = new_breakpoints
    return breakpoints

def convert_breakpoints_empty( occupied_sites, max_complexity):
    full_list = [i for i in range(1,max_complexity+1)]
    remaining_sites=[]
    for x in full_list:
        if x in occupied_sites:
            continue
        else:
            remaining_sites.append(x)
    return remaining_sites


def generate_neighbour(config, max_complexity):
    '''
    Generates a neighbouring configuration from the current one.
    Works both for the breakpoints or the "empty-sites".
    '''
    # store new configuratioin in a copy of the array (to avoid problems with overwriting arrays):
    new_config = config[:]

    # spCas9:
    Nguide = 20

    # 0) Make a legal move: Select a breakpoint/empty site that can move in at least one direction:
    allowed_moves = []
    while len(allowed_moves) == 0:
        # 1) select breakpoint to move:
        index = np.random.randint(low=0, high=len(config))
        chosen_one = config[index]
        left_neighbour = chosen_one - 1
        right_neighbour = min(chosen_one + 1, max_complexity)

        # 2) check what moves are available:
        left_open = left_neighbour not in config
        right_open = right_neighbour not in config
        neighbours_open = [left_neighbour * left_open, right_neighbour * right_open]
        allowed_moves = []
        for i in np.nonzero(neighbours_open)[0]:
            allowed_moves.append(neighbours_open[i])

        # 0) if no allowed_moves, lenght-0, try again.
        # Make the above as while-loop, this replaces the loop to induce a legal move

    # 3) Move to one of the available neighbouring sites, with equal weight if both are available:
    index2 = np.random.randint(low=0, high=len(allowed_moves))
    trial_point = allowed_moves[index2]
    new_config[index] = trial_point
    return new_config