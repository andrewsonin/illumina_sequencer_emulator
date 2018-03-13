from functions import MAX_SEED, print_verbose, cmd_parse, disassembler, rfd
from Bio.SeqIO import parse, to_dict
from multiprocessing import Process
from time import time


def main():
    start_time = time()

    parser = cmd_parse()  # Parsing of command line arguments

    with open(parser['file']) as genome:
        fasta_genome = to_dict(parse(genome, 'fasta'))  # Reading genome file

    jobs = parser['jobs']  # Number of processes to parallelize
    fragments = parser['fragments_num']  # Number of fragments to get
    frags_per_core = [fragments // jobs] * (jobs - 1)
    frags_per_core.append(fragments - sum(frags_per_core))  # Number of fragment to get from one process

    dis_file = parser['dis_file']  # Address of empirical distribution file
    emp_dis = rfd(dis_file) if dis_file is not None else None  # Empirical distribution reading
    my_seed = parser['seed']  # Numpy seeding argument

    processes = []  # List of processes to parallelize
    for job, fragments_num in enumerate(frags_per_core):
        seeding = ((my_seed + job) % MAX_SEED if my_seed != -1 else my_seed) if my_seed is not None else None
        # Processing of seeding argument
        processes.append(Process(target=disassembler,
                                 args=(fasta_genome, parser['seq_type'], fragments_num, parser['out_file'],
                                       parser['depth'], parser['read_len'], job, seeding, emp_dis, parser['mean_len'])))
        processes[-1].start()
    for process in processes:
        process.join()
    print_verbose(
        'The program completed disassembling without any errors. Elapsed time={:f}'.format(time() - start_time),
        parser['session_id'], parser['logfile'], parser['verbose'], parser['params'])  # Parameters logging


if __name__ == '__main__':
    main()
