from collections import defaultdict
from copy import copy
stk = []

def knapsack(s, i, r):
    if i == -1 or r == 0 :
        return 0
    if i in dp:
        return dp[i][r]
    if s[i] > r:
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


def knapsack3(i, r):
    while r > S[i]:
        while i >= 0 :
            print((i, r))
            if S[i] > r:
                dp[i][r] = dp[i-1][r]
                print("dp[i][r]: {}".format(dp[i][r]))
                (i, r) = stk.pop()
                break
            stk.append((i, r))
            stk.append((i, r - S[i]))
            r = r - S[i]
            i = i - 1
        print(stk)
        #stk.append((0, r))
        dp[0][r] = 0
        print(r)
        while i < N:
            (ii, rr) = stk.pop() # i-1, r- s[i]
            (ii, rrr) = stk.pop() # i-1, r
            dp[i][r] = max(S[i] + dp[ii][rr], dp[ii][rrr])
            print(dp[i][r])
            i += 1

# dp = {}
# M = 17
# N = 4
# s = [2, 5, 6, 8]
#dp = [[0 for x in range(M + 1)] for x in range(N + 1)]
#print(knapsack(s, N-1, M))


def test(n):
    if n == 0:
        return 0
    return test(n-1)+1

def test_stack(n):
    myStack = []
    result = 0
    while(n > 0):
        myStack.append(n)
        n = n - 1
    myStack.append(0)
    while(myStack):
        result += 1
        myStack.pop()
    return result


print(test(5))
print(test_stack(5))


if __name__ == "__main__":
    input = open("input/a_example.in")
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    # dp = [[0 for x in range(M + 1)] for x in range(N + 1)]
    tmp_dict = defaultdict(lambda : -1)
    dp = defaultdict(lambda : copy(tmp_dict))
    print(knapsack2(S, N-1, M))
    # print(knapsack3(N-1,M))