import json
import os
import signal

from GA import GA

def timer_handler(signum, frame):
    raise Exception("TTL reached!")

signal.signal(signal.SIGALRM, timer_handler)

if __name__ == "__main__":
    with open("conf.json", 'r') as json_conf:
        conf = json.load(json_conf)

    filenames = [
        'a_example.in',     # 0
        'b_small.in',       # 1
        'c_medium.in',      # 2
        'd_quite_big.in',   # 3
        'e_also_big.in',    # 4
    ]

    input_file = os.path.join(
        conf['file']['in_dir'],
        conf['file']['in_filename']
    )
    GA_conf = conf['GA']
    GA_population_size = GA_conf['population_size']
    GA_mutation_rate = GA_conf['mutation_rate']
    GA_crossover_rate = GA_conf['crossover_rate']

    ttl = conf['TTL']
    use_timer = conf['USE_TIMER']
    if use_timer:
        signal.alarm(ttl)

    with open(input_file, 'r') as infile:
        infile = open(input_file)
        M, N = map(int, infile.readline().rstrip().split())
        S = list(map(int, infile.readline().rstrip().split()))

        # print("max = " + str(M) + " ---- N = "+str(N), '\n\n')

        G = GA.GeneticAlgorithm(S, N, M, pop_size=GA_population_size,
                                mutation_rate=GA_mutation_rate, crossover_rate=GA_crossover_rate)
        try:
            G.run()

        except (KeyboardInterrupt, Exception) as e:
            pass

        finally:
            result = [str(i) for i in range(len(S))
                      if G.best_of_the_bests.genes[i]]
            print(len(result))
            print(' '.join(result))
