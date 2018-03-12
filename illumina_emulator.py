from functions import MAX_SEED, print_verbose, cmd_parse, disassembler
from Bio.SeqIO import parse, to_dict
from multiprocessing import Process
from time import time


def main():
    start_time = time()
    file, out_file, mean_len, loss_frac, read_length, err_frac, seq_type, circular, verbose, logfile, my_seed, \
    session_id, jobs, fragment_num, depth, params = cmd_parse()

    with open(file) as genome:
        fasta_genome = to_dict(parse(genome, 'fasta'))

    fragment_nums = [fragment_num // jobs] * (jobs - 1)
    fragment_nums.append(fragment_num - sum(fragment_nums))
    processes = []
    for job, fragment_num in enumerate(fragment_nums):
        seeding = ((my_seed + job) % MAX_SEED if my_seed != -1 else my_seed) if my_seed is not None else None
        processes.append(Process(target=disassembler,
                                 args=(fasta_genome, seq_type, mean_len, fragment_num, out_file, depth, read_length,
                                       job, seeding)))
        processes[-1].start()
    for process in processes:
        process.join()
    print_verbose(
        'The program completed disassembling without any errors. Elapsed time={:f}'.format(time() - start_time),
        session_id, logfile, verbose, params)


if __name__ == '__main__':
    main()
