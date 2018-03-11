from time import time
from argparse import ArgumentParser
from os.path import isfile
from numpy import logical_and
from numpy.random import random_integers, seed
from multiprocessing import cpu_count
from pdf import ModifiedExponential

MAX_SEED = 4294967295  # 2**32 - 1


class Profiler(object):
    def __enter__(self):
        self._startTime = time()

    def __exit__(self, typ, value, traceback):
        print('Elapsed time: {:.3f} sec'.format(time() - self._startTime))


def get_time_dependent_int(): return int(''.join(str(time()).split('.')))


def print_verbose(string='', session_id=0, logfile='', verbose=False, params=None):
    if verbose:
        with open(logfile, 'a', encoding='utf8') as out:
            out.write('Session_ID={:d}\nTime={:f}\nParams: {:s}\n{:s}\n\n'
                      .format(session_id, time(), str(params), string))
    print(string)


def cmd_parse():
    parser = ArgumentParser(description='Sequencer emulator')

    parser.add_argument('input_file',
                        metavar='INPUT_FILE',
                        type=str,
                        nargs=1,
                        help='Name or address of .fasta-file containing the genome to be disassembled')

    parser.add_argument('-m', '--mean', type=float, action='store', default=[1000], nargs=1,
                        help='Mean length of fragments. Default is 1000')
    parser.add_argument('-l', '--loss', type=float, action='store', default=[0.2], nargs=1,
                        help='Fraction of fragments to lose. Default is 0.2')
    parser.add_argument('-r', '--read', type=int, action='store', default=[150], nargs=1,
                        help='Read length. Default is 150')
    parser.add_argument('-e', '--err', type=float, action='store', default=[0], nargs=1,
                        help='Fraction of erroneous nucleotides. Default is 0')  # FIXME!!! Is not implemented
    parser.add_argument('-t', '--type', type=str, action='store', default=['pe'], nargs=1,
                        help='Type of sequencing. Argument \'pe\' means pair-end sequencing, \'sr\' means singe-read '
                             'one. Passing other arguments raises an error. If not given, the program emulates \'pe\'')
    parser.add_argument('-c', '--circular', action='store_true',
                        help='Is the genome of interest circular')  # FIXME!!! Is not implemented
    parser.add_argument('-v', '--verbose', action='store_true', help='Logging')
    parser.add_argument('-lg', '--log', type=str, action='store', default='log.txt',
                        help='Name or address of log-file. Default is log.txt')
    parser.add_argument('-s', '--seed', type=int, action='store', default=[-1], nargs=1,
                        help='Seed for numpy. Must be between 0 and 2**32 - 1. Default is computed using global time')
    parser.add_argument('-o', '--out', type=str, action='store',
                        help='Name of output file. Default is name of input + disassemble.fastq')
    parser.add_argument('-j', '--jobs', type=int, action='store', default=[1], nargs=1,
                        help='Number of processes to parallelize. Should be between 1 and the number of CPUs. If -1, '
                             'all the CPUs will be engaged. Default is 1')
    parser.add_argument('-n', '--size', type=int, action='store', default=[500], nargs=1,
                        help='Number of reads per iteration. Default is 500')
    parser.add_argument('-i', '--iter', type=int, action='store', default=[5], nargs=1,
                        help='Number of iterations. Default is 5')
    parser.add_argument('-w', '--no_seed', action='store_true', help='Without any seeding including random internal')

    args = parser.parse_args()
    params = {'parser_info': args}
    session_id = get_time_dependent_int()

    if args.seed[0] is not None and args.seed[0] != -1 and not 0 <= args.seed[0] <= MAX_SEED:
        print_verbose('ERROR! Seed must be between 0 and 2**32 - 1', session_id, args.log, args.verbose, params)
        raise ValueError
    elif args.no_seed:
        args.seed[0] = None

    file = args.input_file[0]
    if not isfile(file):
        file += '.fasta'
        if not isfile(file):
            print_verbose('ERROR! No such file: ' + file, session_id, args.log, args.verbose, params)
            raise FileNotFoundError

    if args.type[0] not in ['pe', 'sr']:
        print_verbose('ERROR! --type={:s} is not allowed. Specify it only as \'pe\' or \'sr\''.format(args.type[0]),
                      session_id, args.log, args.verbose, params)
        raise ValueError

    if not 0 <= args.err[0] <= 1:
        print_verbose('ERROR! The value of --err should be between 1 and number of CPUs.', session_id, args.log,
                      args.verbose, params)
        raise ValueError

    if args.out is not None:
        out_file = args.out
    else:
        out_file = ('.'.join(''.join(file).split('.')[:-1]) + 'disassemble_{:s}_id{:d}.fastq')\
            .format(args.type[0], session_id)

    if args.jobs[0] == -1:
        jobs = cpu_count()
    elif 1 <= args.jobs[0] <= cpu_count():
        jobs = args.jobs[0]
    else:
        print_verbose('ERROR! The value of --jobs should be between 0 and the number of CPUs or -1', session_id,
                      args.log, args.verbose, params)
        raise ValueError

    return file, out_file, args.mean[0], args.loss[0], args.read[0], args.err[0], args.type[0], args.circular, \
           args.verbose, args.log, args.seed[0], session_id, jobs, args.size[0], args.iter[0], params


def generate_samples(distribution, len_of_seq, fragment_num):
    samples = distribution.rvs(size=fragment_num) * len_of_seq
    return list(map(int, samples[logical_and(0 < samples, samples < len_of_seq)]))


def get_fragments(seq, samples):
    seq_len = len(seq)
    cleavage_sites = random_integers(0, seq_len - 1, len(samples))
    return [seq[site:end] for site, end in zip(cleavage_sites, cleavage_sites + samples) if end < seq_len]


def get_reads(fragments, length):
    return [(fragment[:length], fragment[-length:]) for fragment in fragments]


def out_write(file, desc, iteration, seq_type, reads, thread, my_seed):
    if seq_type == 'pe':
        number_of_mates = 2
    else:
        number_of_mates = 1
    description = '@{:s} iter={:d} thread={:d} seed={:d}'.format(desc, iteration, thread, my_seed)
    with open(file, 'a', encoding='utf8') as out:
        for number, read in enumerate(reads):
            for i in range(number_of_mates):
                out.write((description + ' read={:d}.{:d}\n{:s}\n+\n').format(number, i, str(read[i])))
                out.write('G' * len(read[i]) + '\n')  # FIXME!!! Error distribution is not implemented


def disassembler(fasta_genome, seq_type, mean_len, fragment_num, out_file, depth_of_seq, read_length, thread, my_seed):
    if my_seed == -1:
        my_seed = (thread + get_time_dependent_int()) % MAX_SEED
    if my_seed is not None:
        seed(my_seed)
    for identifier, info in fasta_genome.items():
        len_of_seq = len(info.seq)
        distribution = ModifiedExponential(len_of_seq / mean_len / 2, len_of_seq, name='ModifiedExponential')
        for iteration in range(depth_of_seq):
            out_write(out_file, info.description, iteration, seq_type,
                      get_reads(get_fragments(info.seq, generate_samples(distribution, len_of_seq, fragment_num)),
                                read_length), thread, my_seed)
