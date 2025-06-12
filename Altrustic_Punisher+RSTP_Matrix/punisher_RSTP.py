import random
import numpy as np
import sys

def assign_initial_strategies(m, n):  # Should give an upper triangular matrix
    grids = np.zeros((n + 1, n + 1))
    # Generate a list of all valid (i, j) pairs for the upper-triangular region.
    valid_indices = [(i, j) for i in range(n + 1) for j in range(n - i + 1)]
    for _ in range(m):
        i, j = random.choice(valid_indices)
        grids[i][j] += 1
    return grids

def draw_time_poisson(leaving_rate_matrix):
    lambd = np.sum(leaving_rate_matrix)
    return np.random.poisson(lambd)

def draw_random_coord(matrix):
    prob_matrix = matrix / np.sum(matrix)
    flat_probs = prob_matrix.flatten()  
    chosen_idx = np.random.choice(len(flat_probs), p=flat_probs)
    return np.unravel_index(chosen_idx, prob_matrix.shape)  # Convert back to (row, col)

def frac(i, j, m, grids):
    return grids[i,j] / m * 1.0

def rou(x, y, u, v, payoff_matrix): 
    #probability of the random (x,y) group wins the pairwise competition, 
    # i.e rou((x,y),(u,v))   x, y are fractions of cooperators and defectors in the group, respectively
    pi_c_xy, pi_d_xy, pi_p_xy = get_coop_payoff(payoff_matrix, x, y), get_def_payoff(payoff_matrix, x, y), get_coop_payoff(payoff_matrix, x, y)
    pi_c_uv, pi_d_uv, pi_p_uv = get_coop_payoff(payoff_matrix, u, v), get_def_payoff(payoff_matrix, u, v), get_coop_payoff(payoff_matrix, u, v)

    Gwin = x * pi_c_xy + y * pi_d_xy + (1 - x - y) * pi_p_xy
    Glose = u * pi_c_uv + v * pi_d_uv + (1 - u - v) * pi_p_uv

    s = 0.1  # s is a non-negative parameter governing the sensitivity of group-level victory probability to the difference in average payoffs of the two competing groups.
    return 0.5 + 0.5 * np.tanh(s * (Gwin - Glose))   


#x and y, in the followings, are fractions of cooperators and defectors in the group, respectively
def get_coop_payoff(payoff_matrix, x, y):
    R, S = payoff_matrix[0][0], payoff_matrix[0][1]
    pi_c = R * x + S * y + R * (1 - x - y)   # Payoff for cooperators
    return pi_c

def get_def_payoff(payoff_matrix, x, y):
    T, P, T_minus_delta = payoff_matrix[1][0], payoff_matrix[1][1], payoff_matrix[1][2]
    pi_d = T * x + P * y + T_minus_delta * (1 - x - y)  # Payoff for defectors
    return pi_d

def get_pun_payoff(payoff_matrix, x, y):
    R, S_minus_epsilon = payoff_matrix[2][0], payoff_matrix[2][1]
    pi_p = R * x + S_minus_epsilon * y + R * (1 - x - y)  # Payoff for punishers
    return pi_p


def leaving_rate(m, n, i, j, pi_c, pi_d, pi_p, payoff_matrix, grids, big_lambda, wi):
    #Group level:
    gl = 0
    for d in range(n+1):
        for l in range(n+1-d):
            if (d == i and l == j):
                continue
            fdl = frac(d, l, m, grids)
            gl += rou(d/n, l/n, i/n, j/n, payoff_matrix) * fdl

    gl *= big_lambda * m * frac(i,j,m,grids)
    #Absorbing state i.e one strategy dominates, we only can change from group-level activity
    if (i == 0 and j == 0) or (j == n) or (i == n):
        return gl

    #Individual level:
    #Cooperator reproduces & punisher dies
    fij = frac(i, j, m, grids)
    cp = m * fij * i * (1 + wi * pi_c) * ((n-i-j)/ n)
    #Punisher reproduces & Cooperator dies
    pc = m * fij * (n - i - j) * (1 + wi * pi_p) * (i / n)
    #Defector reproduces & Punisher dies
    dp = m * fij * j * (1 + wi * pi_d) * ((n-i-j) / n)
    #Punisher reproduces & Defector dies
    pd = m * fij * (n - i - j) * (1 + wi * pi_p) * (j / n)
    #Cooperator reproduces & Defector dies
    cd = m * fij * i * (1 + wi * pi_c) * (j / n)
    #Defector reproduces & Cooperator dies
    dc = m * fij * j * (1 + wi * pi_d) * (i / n)

    #individual level + group level
    li = cp + pc + dp + pd + cd + dc + gl
    return li

def get_total_leaving_rate(m, n, grids, payoff_matrix, big_lambda, wi):
    leaving_rate_matrix = np.zeros_like(grids)
    for i in range(n + 1):
        for j in range(n - i + 1):
            pi_c = get_coop_payoff(payoff_matrix, i/n, j/n)
            pi_d = get_def_payoff(payoff_matrix, i/n, j/n)
            pi_p = get_pun_payoff(payoff_matrix, i/n, j/n)
            lr = leaving_rate(m, n, i, j, pi_c, pi_d, pi_p, payoff_matrix, grids, big_lambda, wi)
            leaving_rate_matrix[i,j] = lr

    return leaving_rate_matrix


def get_total_incoming_rate(m, n, lij, payoff_matrix, big_lambda, wi, grids):
    incoming_rate_matrix = np.zeros_like(grids)
    i = lij[0]
    j = lij[1]

    fij = frac(i, j, m, grids)
    if i < n:
        #(u,v) == (i + 1, j) cooperator reproduces & punisher dies
        indi = i * (1 + wi * get_coop_payoff(payoff_matrix, i/n, j/n)) * (n - i - j) / n
        grp = big_lambda * fij * m * rou((i + 1)/n, j/n, i/n, j/n, payoff_matrix) * frac(i + 1, j, m, grids)
        incoming_rate_matrix[i + 1, j] = (indi + grp) 

        #(u,v) == (i + 1, j - 1) cooperator reproduces & defector dies
        if (j > 0):
            indi = i * (1 + wi * get_coop_payoff(payoff_matrix, i/n, j/n)) * j / n
            grp = big_lambda * fij * m * rou((i + 1)/n, (j-1)/n, i/n, j/n, payoff_matrix) * frac(i + 1, j - 1, m, grids)
            incoming_rate_matrix[i + 1, j - 1] = (indi + grp)


    if i > 0:
        #(u,v) == (i - 1, j) punisher reproduces & cooperator dies
        indi = (n-i-j) * (1 + wi * get_pun_payoff(payoff_matrix, i/n, j/n)) * i / n
        grp = big_lambda * fij * m * rou((i - 1)/n, j/n, i/n, j/n, payoff_matrix) * frac(i - 1, j, m, grids)
        incoming_rate_matrix[i - 1, j] = (indi + grp)

        #(u,v) == (i - 1, j + 1) defector reproduces & cooperator dies
        indi = j * (1 + wi * get_def_payoff(payoff_matrix, i/n, j/n)) * i / n
        grp = big_lambda * fij * m * rou((i - 1)/n, (j+1)/n, i/n, j/n, payoff_matrix) * frac(i - 1, j + 1, m, grids)
        incoming_rate_matrix[i - 1, j + 1] = (indi + grp)

    if j < n:
        #(u,v) == (i, j + 1) defector reproduces & punisher dies
        indi = j * (1 + wi * get_def_payoff(payoff_matrix, i/n, j/n)) * (n - i - j) / n
        grp = big_lambda * fij * m * rou(i/n, (j+1)/n, i/n, j/n, payoff_matrix) * frac(i, j + 1, m, grids)
        incoming_rate_matrix[i, j + 1] = (indi + grp)

    if j > 0:   
        #(u,v) == (i, j - 1) punisher reproduces & defector dies
        indi = (n-i-j) * (1 + wi * get_pun_payoff(payoff_matrix, i/n, j/n)) * j / n
        grp = big_lambda * fij * m * rou(i/n, (j-1)/n, i/n, j/n, payoff_matrix) * frac(i, j - 1, m, grids)
        incoming_rate_matrix[i, j - 1] = (indi + grp)

    #Group level:
    for d in range(n + 1):
        for l in range(n - d + 1):
            if abs(d - i) > 1 or abs(l - j) > 1 or (d == i + 1 and l == j + 1) or (d == i - 1 and l == j - 1):
                incoming_rate_matrix[d,l] = (big_lambda * fij * m * rou(d/n, l/n, i/n, j/n, payoff_matrix) * frac(d,l,m,grids))
            
    return incoming_rate_matrix
        



def main():
    m = 20   # Number of groups 
    n = 3    # Number of individuals in each group
    
    # Define parameters
    R, S, T, P = 1, -1, 2, 0     # Payoff values for the game
    epsilon = 10                 # The cost of punishers to punish. 
    delta = .5                   # The punishment to defector from punisher
    payoff_matrix = [[R, S, R], [T, P, T - delta], [R, S - epsilon, R]]  # Payoff matrix for the game
    big_lambda = 0.1            # rate of group-level competition in which each group engages in pairwise competitions with other groups 
    wi = 0.1                    # Intensity of individual-level selection with respect to the payoff
    
    grids = assign_initial_strategies(m, n) #row represents i, the # of coop; column represents j, the # of defectors; [0,n] for i and j
    print("\033[1;32mEqually distributed grids:\033[0m")
    print(grids)

    while grids[0,0] != m and grids[0,n] != m and grids[n,0] != m:
        leaving_rate_matrix = get_total_leaving_rate(m, n, grids, payoff_matrix, big_lambda, wi)
        if np.any(leaving_rate_matrix < -1e-8):
            raise ValueError(f"\033[31mNegative values in leaving rates (leaving_rate_matrix)\033[0m\n: {leaving_rate_matrix}")

        if np.sum(leaving_rate_matrix) == 0:
            raise ValueError("\033[31mSum of leaving rates (L) is zero, unable to draw random numbers.\033[0m")
            
        tau = draw_time_poisson(leaving_rate_matrix)
        lij = draw_random_coord(leaving_rate_matrix) #the coordinate of the individual who will leave the group (i,j)
        print("\033[1;32mleaving_rate_matrix:\033[0m")
        print(grids)

        incoming_rate_matrix = get_total_incoming_rate(m, n, lij, payoff_matrix, big_lambda, wi, grids)

        if np.any(incoming_rate_matrix < -1e-8):
            raise ValueError(f"\033[31mNegative values in incoming rates (incoming_rate_matrix)\033[0m\n: {incoming_rate_matrix}")
        
        if np.sum(incoming_rate_matrix) == 0:
            raise ValueError("\033[31mSum of incoming rates (L) is zero, unable to draw random numbers.\033[0m")
        
        lij_prime = draw_random_coord(incoming_rate_matrix)

        # Update the grids
        grids[lij] -= 1
        grids[lij_prime] += 1
        print("\033[1;32mUpdated grids:\033[0m")
        print(grids)

        if np.any(grids < 0):
            raise ValueError(f"\033[31mNegative values in grids after iteration:\033[0m\n{grids}")
        
        T += tau

    
    print("\033[1;32mSimulation is done.\033[0m") 
        
    
if __name__ == "__main__":
    main()
