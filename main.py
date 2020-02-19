import threading
from queue import Queue
from slice import GeneticAlgorithm as GA


NUMBER_OF_THREADS = 6
global S, N, M
POP_SIZE = 50
# Thread Queue
queue = Queue()     # Queue of jobs

# Create a worker threads (will die when main exits)
def create_workers():
    for _ in range(NUMBER_OF_THREADS):
        t = threading.Thread(target = work)
        # t.daemon = True     # die when main exits
        t.start()


def release_queue():
    print("releasing queue")
    print(queue.qsize())
    while(queue.qsize()):
        print(queue)
        queue.task_done()

# Do the next job in the queue
def work():
    while True:
        ga = queue.get()
        best = ga.run(threading.current_thread().name)
        print(best.fitness[2])
        queue.task_done()       # Job is done
        release_queue()
        return
# For each GA create new job
def run():
    for x in range(NUMBER_OF_THREADS):
        queue.put(GA(S, N, M, pop_size=POP_SIZE))  # Adding job to job queue
    # queue.join()

if __name__ == "__main__":
    filenames = [
        'a_example.in',     # 0
        'b_small.in',       # 1
        'c_medium.in',      # 2
        'd_quite_big.in',   # 3
        'e_also_big.in',    # 4
    ]

    input = open("input/" + filenames[2])
    M, N = map(int,input.readline().rstrip().split())
    S = list(map(int, input.readline().rstrip().split()))
    print("max = " + str(M) + " ---- N = "+str(N), '\n\n')

    create_workers()
    run()

