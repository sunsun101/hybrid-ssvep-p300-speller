import math

def calculate_itr(N, P, T):
    itr = (math.log2(N) + P * math.log2(P) + (1 - P) * math.log2((1 - P) / (N - 1))) * 60/T
    return itr

# Example usage:
N = 8
T = 4

P_list = [88.00
,81.00
,68.75
,56.25
,68.75
,93.75
,93.75
,87.50
,99.99
,91.00
,83.33
,87.50
,87.50
]


for P in P_list:
    itr_value = calculate_itr(N, P/100, T)
    print("ITR:", itr_value)

    