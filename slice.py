from collections import defaultdict
from copy import copy
import random
import copy
import numpy
stk = []
dp = []

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


# dp = {}
# M = 17
# N = 4
# s = [2, 5, 6, 8]
#dp = [[0 for x in range(M + 1)] for x in range(N + 1)]

class Individual:
    def __init__(self, ln):
        self.length = ln
        self.genes = []
        self.fitness = [0,0] # [score,flag]     flag = -1 => sum > Max

    def random_individual(self):
        self.genes = [random.randint(0, 1) for _ in range(self.length)]

    def calc_fitness(self, slices, Mx):
        val = sum([slices[i] * self.genes[i] for i in range(self.length)])

        fitness = [1/(abs(Mx - val) + 1), 1, val]
        if val > Mx:
            fitness[1] = -1

        self.fitness = fitness

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]


class Population:

    def __init__(self, ln, sz=100):
        self.individuals = []
        self.size = sz
        self.ln = ln

    def add_random_individual(self):
        i = Individual(self.ln)
        i.random_individual()
        self.individuals.append(i)

    def generate_popluation(self):
        for _ in range(self.size):
            self.add_random_individual()


class GeneticAlgorithm:

    def __init__(self, S, N, M, selection_size=100):
        self.slices = S
        self.N_slices = N
        self.Mx = M

        self.population = Population(self.N_slices)
        self.population.generate_popluation()

        self.selected = []
        self.selection_size = selection_size

        self.next_generation = []
        self.best_of_the_bests = Individual(self.N_slices)
        self.best_of_the_bests.random_individual()

    def cross_over(self, instance_a, instance_b, point=0.5):
        instance_a_cpy = copy.copy(instance_a)
        instance_b_cpy = copy.copy(instance_b)

        loc = int(self.N_slices * point)
        instance_a_cpy.genes[:loc], instance_b_cpy.genes[:loc] = instance_b_cpy.genes[:loc],  instance_a_cpy.genes[:loc]

        return instance_a_cpy, instance_b_cpy

    def mutate(self, individual, chance=0.15):
        individual_tmp = copy.copy(individual)
        for i in range(individual_tmp.length):
            ch = random.random()
            if ch <= chance:
                individual_tmp.genes[i] ^= 1
        return individual_tmp

    def find_scores(self):
        for i in range(self.population.size):
            self.population.individuals[i].calc_fitness(self.slices, self.Mx)

    def select(self):
        self.selected = []
        my_sum = sum([i.fitness[0] for i in self.population.individuals])
        selection_probs = [i.fitness[0]/my_sum for i in self.population.individuals]
        # print(my_sum, len(selection_probs), self.population.size)
        self.selected = [self.population.individuals[numpy.random.choice(a=self.population.size, p=selection_probs)] for _ in range(self.selection_size)]

    def do_crossover_and_mutation(self):
        self.next_generation = []
        for i in range(0, self.selection_size, 2):
            a, b = self.cross_over(self.selected[i], self.selected[i+1])
            self.mutate(a)
            self.mutate(b)
            a.calc_fitness(self.slices, self.Mx)
            # print(a.genes, a.fitness)
            b.calc_fitness(self.slices, self.Mx)
            # print (b.genes, b.fitness)
            self.next_generation.append(a)
            self.next_generation.append(b)

    def update_population(self):
        self.next_generation += self.population.individuals
        self.next_generation.sort(reverse=True)
        # print("update pop")
        # for n in self.next_generation:
        #     print(n.genes, n.fitness)
        # self.population.individuals.sort(reverse=True)

        self.population.individuals = self.next_generation[:self.selection_size]

    def run(self):
        self.find_scores()                      # Init: To determine if an instance is good enough
        cnt = 0
        while True:
            self.select()                       # To select the good instances for next generation creation
            self.do_crossover_and_mutation()    # Generate next generation
            self.update_population()            # Copy the newly generated population over the previous population
            self.find_scores()  # To determine if an instance is good enough

            for i in self.population.individuals:
                if i.fitness[1] > 0 and self.best_of_the_bests.fitness[0] < i.fitness[0]:
                    self.best_of_the_bests = i

            if cnt % 27 == 0:
                print(cnt, self.best_of_the_bests.fitness, '---', self.best_of_the_bests.genes)
            cnt += 1


if __name__ == "__main__":
    input = open("input/b_small.in")
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    print("max = " + str(M) + " ---- N = "+str(N))
    # dp = [[0 for x in range(M + 1)] for x in range(N + 1)]
    # tmp_dict = defaultdict(lambda : -1)
    # dp = defaultdict(lambda : copy(tmp_dict))
    # print(knapsack(S, N-1, M))
    G = GeneticAlgorithm(S, N, M)
    G.run()
    tmp_dict = defaultdict(lambda : -1)
    dp = defaultdict(lambda : copy(tmp_dict))
    print(knapsack(S, N-1, M))
    print(dp)
    # print(knapsack3(N-1,M))