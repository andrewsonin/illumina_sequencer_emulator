from scipy.stats import rv_continuous
from numpy.random import choice, randint
from numpy import exp, fromiter


class ModifiedExponential(rv_continuous):  # p(x) = exp(-C) + C exp(-Cx) if x \in [0;1] else 0
    def __init__(self, concentration=None, genome_size=None, *args, **kwargs):
        super().__init__(a=0, *args, **kwargs)
        self.concentration = self.supportive_val = self.genome_size = None
        if concentration is not None:
            self.reload_parameters(concentration, genome_size)

    def reload_parameters(self, concentration, genome_size):
        self.concentration = concentration
        self.genome_size = genome_size
        self.supportive_val = concentration / genome_size

    def _pdf(self, x, *args):
        return self.concentration * exp(-x * self.supportive_val)  # drop the term exp(-C)

    def rls(self, size=1):  # Random lengths
        return self.rvs(size=size) * self.genome_size


class EmpiricalPDF():
    def __init__(self, lengths, probabilities=None):
        self.lengths = lengths
        self.probabilities = probabilities

    def rls(self, size=1):  # Random lengths
        return fromiter((randint(lowest, highest + 1) for lowest, highest in
                         [self.lengths[i] for i in choice(len(self.lengths), size, p=self.probabilities)]), dtype=int)
