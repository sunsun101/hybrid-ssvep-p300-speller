import math

def calculate_itr(N, P, T):
    itr = (math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))) * (60/T)
    return itr

# Example usage:
N = 16
T = 6

P_list = [59.37
]


for P in P_list:
    itr_value = calculate_itr(N, P/100, T)
    print(itr_value)

    