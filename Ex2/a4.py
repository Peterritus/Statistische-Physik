import numpy as np
from numpy import random as rand
import matplotlib.pyplot as plt
from a3 import generate_end_to_end_dist
from a3 import rand_walk


# used to sample with a bias
def dir_choice(bias=0.5):
    if rand.rand() < bias:
        return -1
    else:
        return 1


def is_in_arr(a, b):
    is_in_b = False
    for i in range(len(b[0])):
        if np.all(b[:, i] == a):
            is_in_b = True
            # print(b[:, i], a)
        # else:
            # print('they are not equal',  b[:, i], a)

    # print('returned', is_in_b)
    return is_in_b


# returns an array of dim d, N that shows all steps of a rw
def rand_walk_wo_backtrack(N: int, d: int, bias=0.5):
    rw = np.zeros((d, N))
    last_dir = np.zeros(2)
    for i in range(1, N):
        rw[:, i] = rw[:, i - 1]

        add_at = rand.randint(0, d)
        direction = dir_choice(bias=bias)

        while last_dir[0] == add_at and last_dir[1] == -direction:
            add_at = rand.randint(0, d)
            direction = dir_choice(bias=bias)

        rw[add_at, i] += direction
        last_dir = add_at, direction
    return rw


def rand_walk_wo_intersect(N: int, d: int):
    rw = np.zeros((d, N))

    for i in range(1, N):
        rw[:, i] = rw[:, i - 1]
        dir_choices = [(x % d, 1 if x // d & 1 else -1) for x in range(2 * d)]
        is_free = False
        proposed_step = np.zeros(d)
        early_stop = False
        while len(dir_choices) != 0 and is_free is False:

            r = rand.randint(len(dir_choices))
            choice = dir_choices.pop(r)
            dir_arr = np.zeros(d)
            dir_arr[choice[0]] += choice[1]
            proposed_step = rw[:, i] + dir_arr
            is_in_rw = is_in_arr(proposed_step, rw)
            if len(dir_choices) == 0 and is_in_rw:
                print('got stuck')
                print(i)
                early_stop = True
                rw = rw[:, 0:i]
                break
            if not is_in_rw:
                is_free = True
        if early_stop:

            break
        rw[:, i] = proposed_step
    # deliberately not returning the length here, since it is returned in the rw array shape anyway.
    # this way all rw functions return the same objects and types
    return rw


if __name__ == "__main__":
    #######
    # 4.1 #
    #######
    rw = rand_walk_wo_backtrack(10, 2)

    fig1, ax1 = plt.subplots()
    distances = rand_walk_wo_backtrack(1000, 2)
    ax1.plot(distances[0], distances[1], label='rw w/o backtrack')
    ax1.set_xlabel('x direction')
    ax1.set_ylabel('y direction')
    fig1.savefig('a4_1_rw_without_backtrack.pdf')

    #######
    # 4.2 #
    #######

    # plotting
    fig2, ax2 = plt.subplots()
    # plot without backtrack
    distances = generate_end_to_end_dist(1000, 2, stepsize=10, n_runs=100, rw_func=rand_walk_wo_backtrack)
    ax2.plot(distances[0], distances[1], label='end to end distances for rw w/o backtrack')
    # linear fit
    fit = np.polyfit(distances[0], distances[1], 1)
    best_fit_line = np.poly1d(fit)
    ax2.plot(distances[0], best_fit_line(distances[0]), label='linear fit w/o bt')

    # plot with option of going back
    distances = generate_end_to_end_dist(1000, 2, stepsize=10, n_runs=100)
    ax2.plot(distances[0], distances[1], label='end to end distances for rw with backtrack')
    fit = np.polyfit(distances[0], distances[1], 1)
    best_fit_line = np.poly1d(fit)
    ax2.plot(distances[0], best_fit_line(distances[0]), label='linear fit with bt')

    # general plotting
    ax2.set_xlabel('Number of Steps')
    ax2.set_ylabel('Average Distance')
    ax2.set_title('Average Distance vs. Number of Steps averaged over 100 runs')
    plt.legend()
    fig2.savefig('a4_2_distances_rw_wo_backtrack.pdf')
    plt.show()

    #######
    # 4.3 #
    #######

    fig3, ax3 = plt.subplots(3, 2)
    n_full_rws = 0
    n_tries = 0
    while n_full_rws < 6:
        n_tries += 1
        rw = rand_walk_wo_intersect(50, 2)
        print(len(rw[0]))
        if len(rw[0]) == 50:
            ax3[n_full_rws // 2, n_full_rws % 2].plot(rw[0], rw[1], label='rw w/o intersection')
            ax3[n_full_rws // 2, n_full_rws % 2].set_xlabel('x distance')
            ax3[n_full_rws // 2, n_full_rws % 2].set_ylabel('y distance')
            ax3[n_full_rws // 2, n_full_rws % 2].set_title(f'Try number {n_tries}')
            n_full_rws += 1

    fig3.suptitle(f'random walks with n = 50, took {n_tries} tries to generate')
    plt.tight_layout()
    # fig3.set_title()
    fig3.savefig('a4_3_rw_without_intersection.pdf')
    print(f'number of tries was {n_tries}')
