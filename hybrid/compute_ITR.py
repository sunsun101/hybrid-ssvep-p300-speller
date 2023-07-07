import math

def calculate_itr(N, P, T):
    itr = (math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))) * (60/T)
    return itr

# Example usage:
N = 8
T = 4

P_list = [83.33
,58.33
,50
,66.66
,83.33
,72.22
,97.2
,77.77
,61.11
,47.77
,75
,58.33
,86
]


for P in P_list:
    itr_value = calculate_itr(N, P/100, T)
    print(itr_value)

    