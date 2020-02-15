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
    if dp[i][r] >= 0:
        return dp[i][r]
    if s[i] > r:
        dp[i][r] = knapsack(s, i-1, r)
        return dp[i][r]
    dp[i][r] = max(s[i] + knapsack(s, i-1, r - s[i]), knapsack(s, i-1, r))
    return dp[i][r]


def print_items(s, n, m, result):
    picked_items = [0 for _ in range(n+1)]
    for i in range(n, -1, -1):
        if result <= 0:
            break
        if result == dp[i-1][m]:
            picked_items[i] = 0
            continue
        else:
            picked_items[i] = 1
            m = m - s[i]
            result = result - s[i]
    print(picked_items)

class Individual:
    def __init__(self, ln):
        self.length = ln
        self.genes = []
        self.fitness = [0, 0, 0]    # [score,flag]     flag = -1 => sum > Max

    def random_individual(self):
        self.genes = [random.randint(0, 1) for _ in range(self.length)]

    def calc_fitness(self, slices, maximum_slices):
        val = sum([slices[i] * self.genes[i] for i in range(self.length)])
        fitness = [1/(abs(maximum_slices - val) + 1), 1, val]
        if val > maximum_slices:
            fitness[1] = -1

        self.fitness = fitness

    def __lt__(self, other):
        if self.fitness[0] == other.fitness[0]:
            return self.fitness[1] > other.fitness[1]
        return self.fitness[0] < other.fitness[0]

    def __eq__(self, other):
        if self.fitness == other.fitness:
            if self.genes == other.genes:
                return True
        return False

    def __str__(self):
        return str(self.genes)+"\n"

    def __repr__(self):
        return str(self.genes)+"\n"

class Population:
    def __init__(self, ln, sz=100):
        self.individuals = []
        self.size = sz
        self.gene_len = ln

    def add_random_individual(self):
        i = Individual(self.gene_len)
        i.random_individual()
        self.individuals.append(i)

    def generate_popluation(self):
        for _ in range(self.size):
            self.add_random_individual()

    def __str__(self):
        result = ""
        for i in self.individuals:
            result += str(i)+"\n"
        return result


class GeneticAlgorithm:

    def __init__(self, S, N, M, pop_size=50):
        self.slices = S
        self.size_of_slices = N
        self.max_slices = M

        self.population = Population(self.size_of_slices, pop_size)
        self.population.generate_popluation()

        self.selected = []

        self.next_generation = copy.copy(self.population.individuals)
        self.best_of_the_bests = Individual(self.size_of_slices)
        self.best_of_the_bests.random_individual()

    def cross_over(self, instance_a, instance_b, point=0.5):
        instance_a_cpy = copy.deepcopy(instance_a)
        instance_b_cpy = copy.deepcopy(instance_b)
        loc = int(self.size_of_slices * point)
        instance_a_cpy.genes[:loc], instance_b_cpy.genes[:loc] = instance_b_cpy.genes[:loc],  instance_a_cpy.genes[:loc]
        return instance_a_cpy, instance_b_cpy

    def mutate(self, individual, chance=0.5):
        individual_tmp = copy.deepcopy(individual)
        for i in range(individual_tmp.length):
            ch = random.random()
            if ch <= chance:
                individual_tmp.genes[i] ^= 1
        return individual_tmp

    def find_scores(self):
        # for i in range(self.population.size):
        #     for j in range(len(self.population.individuals)):
        #         self.population.individuals[j].calc_fitness(self.slices, self.max_slices)
        for i in self.next_generation:
            i.calc_fitness(self.slices, self.max_slices)

    def select(self):
        self.selected = []
        total = sum([i.fitness[0] for i in self.population.individuals])
        selection_probs = [i.fitness[0]/total for i in self.population.individuals]
        self.selected = [self.population.individuals[numpy.random.choice(a=self.population.size, p=selection_probs)] for _ in range(self.population.size)]
        self.next_generation.sort(reverse=True)

    def do_crossover_and_mutation(self):
        self.next_generation = []
        for i in range(0, self.population.size, 2):
            a, b = copy.deepcopy(self.cross_over(self.selected[i], self.selected[i+1]))
            a = copy.copy(self.mutate(a, a.fitness[1]*(1 - a.fitness[2]/self.max_slices)))
            b = copy.copy(self.mutate(b, b.fitness[1] * (1 - b.fitness[2] / self.max_slices)))

            # a = copy.copy(self.mutate(a, 1 - a.fitness[0]))
            # b = copy.copy(self.mutate(b, 1 - b.fitness[0]))

            # a = copy.copy(self.mutate(a))
            # b = copy.copy(self.mutate(b))

            a.calc_fitness(self.slices, self.max_slices)
            b.calc_fitness(self.slices, self.max_slices)
            self.next_generation.append(a)
            self.next_generation.append(b)

    def update_population(self):
        self.next_generation = copy.deepcopy(self.population.individuals) + copy.deepcopy(self.next_generation)
        self.next_generation.sort(reverse=True)
        self.population.individuals = [copy.deepcopy(self.best_of_the_bests)]\
                                      +copy.deepcopy(self.next_generation[:self.population.size - 21])\
                                      +copy.deepcopy(self.next_generation[-20:])

    def run(self):
        self.find_scores()                      # Init: To determine if an instance is good enough
        cnt = 0
        while True:
            self.select()                       # To select the good instances for next generation creation
            self.do_crossover_and_mutation()    # Generate next generation
            # self.find_scores()  # To determine if an instance is good enough
            self.update_population()            # Copy the newly generated population over the previous population

            for i in self.population.individuals:
                if i.fitness[1] > 0 and self.best_of_the_bests.fitness[0] < i.fitness[0]:
                    self.best_of_the_bests = copy.deepcopy(i)
            if cnt % 200 == 0:
                print(cnt, self.best_of_the_bests.fitness, '---', self.best_of_the_bests.genes)
                print(self.next_generation)
            cnt += 1


if __name__ == "__main__":
    input = open("input/c_medium.in")
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    print("max = " + str(M) + " ---- N = "+str(N))

    tmp_dict = defaultdict(lambda: -1)
    dp = defaultdict(lambda: copy.copy(tmp_dict))
    result = knapsack(S, N - 1, M)
    print(result)
    print_items(S, N-1, M, result)


    G = GeneticAlgorithm(S, N, M)
    G.run()

