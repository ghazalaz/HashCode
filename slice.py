from collections import defaultdict
from copy import copy
stk = []

def knapsack(s, i, r):
    if i == -1 or r == 0:
        return 0
    if i in dp:
        return dp[i][r]
    if s[i] >= r:
        dp[i][r] = knapsack(s, i-1, r)
        return dp[i][r]
    dp[i][r] = max(s[i] + knapsack(s, i-1, r - s[i]), knapsack(s, i-1, r))
    return dp[i][r]

def knapsack2(s, i, r):
    stk.append((i, r))

    while len(stk) > 0:
        ii, rr = stk.pop()
        print(ii, rr)
        if ii == -1 or rr == 0 :
            # return 0
            dp[ii][rr] = 0

        if dp[ii][rr] >= 0:
            continue
        stk.append((ii, rr))
        #iii = ii - 1
        if s[ii] > rr:
            stk.append((ii-1, rr))

        # not include s[ii]
        if dp[ii-1][rr]<0:
            stk.append((ii - 1, r))
        # include s[ii]
        if dp[ii-1][rr-s[ii]] < 0:
            stk.append((ii - 1, rr - s[ii]))

        #dp[i][r] = max(s[i] + knapsack(s, i-1, r - s[i]), knapsack(s, i-1, r))

    return dp[i][r]


if __name__ == "__main__":
    input = open("input/b_small.in")
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    # dp = [[0 for x in range(M + 1)] for x in range(N + 1)]
    tmp_dict = defaultdict(lambda : -1)
    dp = defaultdict(lambda : copy(tmp_dict))
    print(knapsack(S, N-1, M))
    print(dp)
    # print(knapsack3(N-1,M))