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

def frac(i, j, m, grids):
    return grids[i,j] / m * 1.0

def rou(x,y,u,v,b,c,q,p,k): #probability of the random (x,y) group wins the pairwise competition, i.e rou((x,y),(u,v))   x, y are fractions of cooperators and defectors in the group, respectively
    s = 0.1 #s is a non-negative parameter governing the sensitivity of group-level victory probability to the difference in average payoffs of the two competing groups.
    Gwin = b - c - q + q * x - (b - c - q + p + k) * y + (p + k) * x * y + (p + k) * (y ** 2) #(x, y) group wins the pairwise competition
    Glose = b - c - q + q * u - (b - c - q + p + k) * v + (p + k) * u * v + (p + k) * (v ** 2)
    return 0.5 + 0.5 * np.tanh(s * (Gwin - Glose))   

def leaving_rate(i, j, pi_c, pi_d, pi_p, wi, m, n, grids, big_lambda, b, c, q, p, k):
    #Group level:
    gl = 0
    for d in range(n+1):
        for l in range(n+1-d):
            if (d == i and l == j):
                continue
            fdl = frac(d, l, m, grids)
            gl += rou(d/n,l/n,i/n,j/n,b,c,q,p,k) * fdl

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

def get_total_leaving_rate(m, n, wi, b, c, p, k, q, grids, big_lambda):
    leaving_rate_matrix = np.zeros_like(grids)
    for i in range(n + 1):
        for j in range(n - i + 1):
            pi_c = get_coop_payoff(b, c, i, j, n)
            pi_d = get_def_payoff(b, p, i, j, n)    
            pi_p = get_pun_payoff(b, c, q, k, i, j, n)
            lr = leaving_rate(i, j, pi_c, pi_d, pi_p, wi, m, n, grids, big_lambda, b, c, q, p, k)
            leaving_rate_matrix[i,j] = lr

    return leaving_rate_matrix

def get_total_incoming_rate(m, n, wi, b, c, p, k, q, grids, big_lamda, lij):
    incoming_rate_matrix = np.zeros_like(grids)
    i = lij[0]
    j = lij[1]

    fij = frac(i, j, m, grids)
    
    if i < n:
        #(u,v) == (i + 1, j) cooperator reproduces & punisher dies
        indi = i * (1 + wi * get_coop_payoff(b, c, i, j, n)) * (n - i - j) / n
        grp = big_lamda * fij * m * rou((i + 1)/n, j/n, i/n, j/n,b,c,q,p,k) * frac(i + 1, j, m, grids)
        incoming_rate_matrix[i + 1, j] = (indi + grp) 

        #(u,v) == (i + 1, j - 1) cooperator reproduces & defector dies
        if (j > 0):
            indi = i * (1 + wi * get_coop_payoff(b, c, i, j, n)) * j / n
            grp = big_lamda * fij * m * rou((i + 1)/n, (j-1)/n, i/n, j/n,b,c,q,p,k) * frac(i + 1, j - 1, m, grids)
            incoming_rate_matrix[i + 1, j - 1] = (indi + grp)
        
    if i > 0:
        #(u,v) == (i - 1, j) punisher reproduces & cooperator dies
        indi = (n-i-j) * (1 + wi * get_pun_payoff(b, c, q, k, i, j, n)) * i / n
        grp = big_lamda * fij * m * rou((i - 1)/n, j/n, i/n, j/n,b,c,q,p,k) * frac(i - 1, j, m, grids)
        incoming_rate_matrix[i - 1, j] = (indi + grp)

        #(u,v) == (i - 1, j + 1) defector reproduces & cooperator dies
        indi = j * (1 + wi * get_def_payoff(b, p, i, j, n)) * i / n
        grp = big_lamda * fij * m * rou((i - 1)/n, (j+1)/n, i/n, j/n,b,c,q,p,k) * frac(i - 1, j + 1, m, grids)
        incoming_rate_matrix[i - 1, j + 1] = (indi + grp)
    
    if j < n:
        #(u,v) == (i, j + 1) defector reproduces & punisher dies
        indi = j * (1 + wi * get_def_payoff(b, p, i, j, n)) * (n - i - j) / n
        grp = big_lamda * fij * m * rou(i/n, (j+1)/n, i/n, j/n,b,c,q,p,k) * frac(i, j + 1, m, grids)
        incoming_rate_matrix[i, j + 1] = (indi + grp)

    if j > 0:   
        #(u,v) == (i, j - 1) punisher reproduces & defector dies
        indi = (n-i-j) * (1 + wi * get_pun_payoff(b, c, q, k, i, j, n)) * j / n
        grp = big_lamda * fij * m * rou(i/n, (j-1)/n, i/n, j/n,b,c,q,p,k) * frac(i, j - 1, m, grids)
        incoming_rate_matrix[i, j - 1] = (indi + grp)

    #Group level:
    for d in range(n + 1):
        for l in range(n - d + 1):
            if abs(d - i) > 1 or abs(l - j) > 1 or (d == i + 1 and l == j + 1) or (d == i - 1 and l == j - 1):
                incoming_rate_matrix[d,l] = (big_lamda * fij * m * rou(d/n, l/n, i/n, j/n,b,c,q,p,k) * frac(d,l,m,grids))
            
    return incoming_rate_matrix

def get_coop_payoff(b, c, i, j, n):
    pi_c = b * (1. - j / n) - c
    return pi_c

def get_def_payoff(b, p, i, j, n):
    pi_d = b * (1. - (j / n)) - p * (1 - (i / n) - (j / n))
    return pi_d

def get_pun_payoff(b, c, q, k, i, j, n):
    pi_c = b * (1. - j / n) - c - q - k * (j / n)
    return pi_c

def draw_time_poisson(leaving_rate_matrix):
    lambd = np.sum(leaving_rate_matrix)
    return np.random.poisson(lambd)

def draw_random_coord(matrix):
    prob_matrix = matrix / np.sum(matrix)
    flat_probs = prob_matrix.flatten()  
    chosen_idx = np.random.choice(len(flat_probs), p=flat_probs)
    return np.unravel_index(chosen_idx, prob_matrix.shape)  # Convert back to (row, col)

def main():
    m = 20   # Number of groups 
    n = 6    # Number of individuals in each group
    
    # Define parameters
    wi = 0.9  # Intensity of individual-level selection
    b = 2.0     # Benefit of cooperation
    c = 1.0     # Cost of benefiting others 
    p = 0   # Additional cost to confer a punishment to a defector
    k = 0     # Fixed cost of punishment for each interaction 
    q = 0   # Single fixed cost to confer punishment on defectors 

    big_lamda = 0.1     #rate of group-level competition in which each group engages in pairwise competitions with other groups 
    
    

    c_vs_p_cwin = 0
    c_vs_p_pwin = 0
    c_vs_p_equal = 0
    
    with open('output.txt', 'w') as f1:
        f1.write("")  # Clear file content at the start
    #TODO run 100 more times on the 100 time to see if punisher wins more or cooperator when we do not include the punishment effect 
    for o in range(100): 
        all_coop = 0
        all_def = 0
        all_pun = 0
        for i in range(100):
            grids = assign_initial_strategies(m, n) #row represents i, the # of coop; column represents j, the # of defectors; [0,n] for i and j
            #print("\033[1;32mEqually distributed grids:\033[0m")
            #print(grids)

            # with open('output.txt', 'w') as f1:
            #     f1.write("")  # Clear file content at the start

            T = 0.0  # time
            
        
            while grids[0,0] != m and grids[0,n] != m and grids[n,0] != m:
                leaving_rate_matrix = get_total_leaving_rate(m, n, wi, b, c, p, k, q, grids, big_lamda)

                if np.any(leaving_rate_matrix < -1e-8):
                    raise ValueError(f"\033[31mNegative values in leaving rates (leaving_rate_matrix)\033[0m\n: {leaving_rate_matrix}")

                if np.sum(leaving_rate_matrix) == 0:
                    raise ValueError("\033[31mSum of leaving rates (L) is zero, unable to draw random numbers.\033[0m")
                
                tau = draw_time_poisson(leaving_rate_matrix)
                lij = draw_random_coord(leaving_rate_matrix) #the coordinate of the individual who will leave the group (i,j)
                incoming_rate_matrix = get_total_incoming_rate(m, n, wi, b, c, p, k, q, grids, big_lamda, lij)

                if np.any(incoming_rate_matrix < -1e-8):
                    raise ValueError(f"\033[31mNegative values in incoming rates (incoming_rate_matrix)\033[0m\n: {incoming_rate_matrix}")
                
                if np.sum(incoming_rate_matrix) == 0:
                    raise ValueError("\033[31mSum of incoming rates (L) is zero, unable to draw random numbers.\033[0m")
                
                lij_prime = draw_random_coord(incoming_rate_matrix)

                # Update the grids
                grids[lij] -= 1
                grids[lij_prime] += 1
                #print("\033[1;32mUpdated grids:\033[0m")
                #print(grids)

                if np.any(grids < 0):
                    raise ValueError(f"\033[31mNegative values in grids after iteration:\033[0m\n{grids}")
                
                
                T += tau
                

            with open('output.txt', 'a') as f1:
                f1.write(f"Iteration {o} Iteration {i} final grid:\n")
                f1.write(str(grids) + "\n")
                f1.flush()
            if grids[0,0] == m:
                all_pun += 1
            elif grids[0,n] == m:
                all_def += 1
            elif grids[n,0] == m:
                all_coop += 1
        # Add your simulation code here

        print("\033[1;32mSimulation is done.\033[0m") 
        # print(f"Cooperator wins: {all_coop}")
        # print(f"Defector wins: {all_def}")
        # print(f"Punisher wins: {all_pun}")  
        if all_coop > all_pun:
            c_vs_p_cwin += 1
        elif all_pun > all_coop:
            c_vs_p_pwin += 1
        elif all_coop == all_pun:
            c_vs_p_equal += 1
    
    print(f"c_vs_p_cwin: {c_vs_p_cwin}")
    print(f"c_vs_p_pwin: {c_vs_p_pwin}")
    print(f"c_vs_p_equal: {c_vs_p_equal}")  
    
if __name__ == "__main__":
    main()
