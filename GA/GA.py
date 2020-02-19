from collections import defaultdict
from copy import copy
import random
import copy
import numpy


class Individual:
    def __init__(self, ln):
        self.length = ln
        self.genes = []
        self.fitness = [0, 0, 0]    # [score,flag]     flag = -1 => sum > Max

    def random_individual(self):
        self.genes = [random.randint(0, 1) for _ in range(self.length)]

    def specific_individual(self, s):
        self.genes = [s for _ in range(self.length)]

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

    def add_specific_individual(self, pattern):
        i = Individual(self.gene_len)
        i.specific_individual(int(pattern))
        self.individuals.append(i)

    def generate_popluation(self, options="r"):
        if 'r' in options:
            for _ in range(self.size):
                self.add_random_individual()
        else:
            for _ in range(self.size):
                self.add_specific_individual(int(options))

    def __str__(self):
        result = ""
        for i in self.individuals:
            result += str(i)+"\n"
        return result


class GeneticAlgorithm:

    def __init__(self, S, N, M, pop_size=50, mutation_rate=0.05, crossover_rate=0.8):
        self.slices = S
        self.size_of_slices = N
        self.max_slices = M
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.population = Population(self.size_of_slices, pop_size)
        self.population.generate_popluation()

        self.selected = []

        self.next_generation = copy.copy(self.population.individuals)
        self.best_of_the_bests = Individual(self.size_of_slices)
        self.best_of_the_bests.random_individual()

    def cross_over(self, instance_a, instance_b, point=0.5, rate=0.8):
        point = random.random()
        if random.random() > rate:
            return instance_a, instance_b
        instance_a_cpy = copy.deepcopy(instance_a)
        instance_b_cpy = copy.deepcopy(instance_b)
        loc = int(self.size_of_slices * point)
        instance_a_cpy.genes[:loc], instance_b_cpy.genes[:
            loc] = instance_b_cpy.genes[:loc],  instance_a_cpy.genes[:loc]
        return instance_a_cpy, instance_b_cpy

    def mutate(self, individual, rate=0.015):
        individual_tmp = copy.deepcopy(individual)
        for i in range(individual_tmp.length):
            ch = random.random()
            if ch <= rate:
                individual_tmp.genes[i] ^= 1
        return individual_tmp

    def find_scores(self, options="n"):
        if 'p' in options:
            for i in range(self.population.size):
                for j in range(len(self.population.individuals)):
                    self.population.individuals[j].calc_fitness(
                        self.slices, self.max_slices)
        if 'n' in options:
            for i in self.next_generation:
                i.calc_fitness(self.slices, self.max_slices)

    def select(self):
        self.selected = []
        total = sum([i.fitness[0] for i in self.population.individuals])
        selection_probs = [i.fitness[0] /
                           total for i in self.population.individuals]
        self.selected = [self.population.individuals[numpy.random.choice(
            a=self.population.size, p=selection_probs)] for _ in range(self.population.size)]
        self.next_generation.sort(reverse=True)

    def do_crossover_and_mutation(self):
        self.next_generation = []
        for i in range(0, self.population.size, 2):
            a, b = copy.deepcopy(self.cross_over(
                self.selected[i], self.selected[i+1], rate=self.crossover_rate))
            if random.random() < 0.2:
                a = copy.copy(self.mutate(
                    a, (a.fitness[1] * (1 - a.fitness[2] / self.max_slices))**(1/2)))
                b = copy.copy(self.mutate(
                    b, (b.fitness[1] * (1 - b.fitness[2] / self.max_slices))**(1/2)))

            else:
                a = copy.copy(self.mutate(a, rate=self.mutation_rate))
                b = copy.copy(self.mutate(b, rate=self.mutation_rate))

            a.calc_fitness(self.slices, self.max_slices)
            b.calc_fitness(self.slices, self.max_slices)

            self.next_generation.append(a)
            self.next_generation.append(b)

    def update_population(self):
        self.next_generation.sort(reverse=True)
        keep_pop = int(self.pop_size * 0.3)
        rand_pop = int(self.pop_size * 0.3)
        # best_of_the_bests and all_1 is also included
        next_pop = self.pop_size - keep_pop - rand_pop - 2

        random.shuffle(self.population.individuals)
        keep_gen = self.population.individuals[:keep_pop]

        rand_ind = Population(self.size_of_slices, rand_pop)
        rand_ind.generate_popluation()
        rand_gen = rand_ind.individuals[:]
        for i in rand_ind.individuals:
            i.calc_fitness(self.slices, self.max_slices)

        all_1 = Population(self.size_of_slices, 1)
        all_1.generate_popluation(options='1')
        all_1_gen = all_1.individuals[:]
        all_1_gen[0].calc_fitness(self.slices, self.max_slices)

        next_gen = self.next_generation[:next_pop]

        self.population.individuals = [copy.deepcopy(
            self.best_of_the_bests)] + keep_gen[:] + next_gen[:] + all_1_gen[:]

        for j in range(len(self.population.individuals)-1, -1, -1):
            if self.population.individuals[j].fitness[1] < 0:
                t_i = self.reduce_individual(copy.deepcopy(
                    self.population.individuals[j]), random.random(), option=1)
                self.population.individuals.append(t_i)

                t_i = self.reduce_individual(copy.deepcopy(
                    self.population.individuals[j]), random.random(), option=0)
                self.population.individuals.append(t_i)

                t_i_2 = self.mutate(self.population.individuals[j], rate=0.05)
                t_i_2.calc_fitness(self.slices, self.max_slices)
                self.population.individuals.append(t_i_2)

        self.population.individuals += rand_gen[:]
        self.population.individuals = self.population.individuals[:self.pop_size]

    def reduce_individual(self, individual, point, option=1):
        if random.random() > 0.5:
            option = 0
        else:
            option = 1
        value = individual.fitness[2]
        if option == 0:
            loc = int(point * self.size_of_slices)
            while loc >= 0 and value > self.max_slices:
                value -= individual.genes[loc] * self.slices[loc]
                individual.genes[loc] = 0
                loc -= 1
        elif option == 1:
            while value > self.max_slices:
                loc = int(random.random() * self.size_of_slices)
                value -= individual.genes[loc] * self.slices[loc]
                individual.genes[loc] = 0
        individual.calc_fitness(self.slices, self.max_slices)
        return individual

    def run(self):
        # Init: To determine if an instance is good enough
        self.find_scores()
        cnt = 0
        while True:
            # To select the good instances for next generation creation
            self.select()
            self.do_crossover_and_mutation()    # Generate next generation
            # Copy the newly generated population over the previous population
            self.update_population()

            for i in self.population.individuals:
                if i.fitness[1] > 0 and self.best_of_the_bests.fitness[0] < i.fitness[0]:
                    self.best_of_the_bests = copy.deepcopy(i)
            if cnt % 20 == 0:
                pass
                # print([i.fitness[2] for i in self.population.individuals])
                # print(cnt, self.best_of_the_bests.fitness)
            cnt += 1
            if self.best_of_the_bests.fitness[0] == 1:
                return True
