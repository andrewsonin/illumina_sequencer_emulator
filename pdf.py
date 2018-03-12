from scipy.stats import rv_continuous
from numpy.random import choice, randint
from numpy import exp, fromiter


class ModifiedExponential(rv_continuous):  # p(x) = exp(-C) + C exp(-Cx) if x \in [0;1] else 0
    def __init__(self, concentration, genome_size, *args, **kwargs):
        super().__init__(a=0, *args, **kwargs)
        self.concentration = concentration
        self.supportive_val = concentration / genome_size

    def _pdf(self, x, *args):
        return self.concentration * exp(-x * self.supportive_val)  # drop the term exp(-C)


class EmpiricalPDF():
    def __init__(self, lengths, probabilities=None):
        self.lengths = lengths
        self.probabilities = probabilities

    def rvs(self, size=1):
        return fromiter((randint(lowest, highest + 1) for lowest, highest in
                         [self.lengths[i] for i in choice(len(self.lengths), size, p=self.probabilities)]), dtype=int)
