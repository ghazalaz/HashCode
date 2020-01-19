from collections import defaultdict
from copy import copy
import random
import copy
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


# dp = {}
# M = 17
# N = 4
# s = [2, 5, 6, 8]
#dp = [[0 for x in range(M + 1)] for x in range(N + 1)]

class Individual:
    def __init__(self, ln):
        self.length = ln
        self.gene = []
        self.fitness = 0

    def random_individual(self):
        self.gene = [random.randint(0, 1) for _ in range(self.length)]

    def calc_fitness(self, slices, Mx):
        val = sum([slices[i] * instance.instance[i] for i in range(self.length)])

        fitness = [abs(Mx - val), 1]
        if val > Mx:
            fitness[1] = -1

        self.fitness = fitness

    def __lt__(self, other):
        pass



class Population:

    def __init__(self, N, sz=100):
        self.people = []
        self.Size = sz
        self.N = N

    def add_random_individual(self):
        i = Individual(self.N)
        i.random_individual()
        self.people.append(i)

    def generate_popluation(self):
        for _ in range(self.Size):
            self.add_random_individual()

class GeneticAlgorithm:

    def __init__(self, S, N, M):
        self.slices = S
        self.N_slices = N
        self.Mx = M

        self.population = Population(self.N_slices)
        self.population.generate_popluation()

    def cross_over(self, instance_a, instance_b, point=0.5):
        instance_a_cpy = copy.copy(instance_a)
        instance_b_cpy = copy.copy(instance_b)

        loc = int(self.N_slices * point)
        instance_a_cpy[:loc], instance_b_cpy[:loc] = instance_b_cpy[:loc],  instance_a_cpy[:loc]

        return instance_a_cpy, instance_b_cpy

    def mutate(self, instance, chance=0.05):
        instance_tmp = copy.copy(instance)
        for i in range(instance_tmp.length):
            ch = random.random()
            if ch <= chance:
                instance_tmp.instance[i] ^= 1
        return  instance_tmp


    def find_scores(self):
        for individual in self.population.people:
            individual.calc_fitness(self.slices, self.Mx)

    def select(self):
        pass

    def update_population(self):
        pass

    def run(self):
        self.find_scores()                      # Init: To determine if an instance is good enough
        tmp_pop = copy.copy(self.population)
        while True:
            self.select()                       # To select the good instances for next generation creation
            self.do_crossover_and_mutation()    # Generate next generation
            self.update_population()            # Copy the newly generated population over the previous population
            self.find_scores()  # To determine if an instance is good enough



if __name__ == "__main__":
    input = open("input/c_medium.in")
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    # dp = [[0 for x in range(M + 1)] for x in range(N + 1)]
    tmp_dict = defaultdict(lambda : -1)
    dp = defaultdict(lambda : copy(tmp_dict))
    print(knapsack(S, N-1, M))
