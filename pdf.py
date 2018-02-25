from scipy.stats import rv_continuous
from numpy import exp


class ModifiedExponential(rv_continuous):  # p(x) = exp(-C) + C exp(-Cx) if x \in [0;1] else 0
    def __init__(self, concentration, genome_size, *args, **kwargs):
        super().__init__(a=0, *args, **kwargs)
        self.concentration = concentration
        self.supportive_val = concentration / genome_size

    def _pdf(self, x, *args):
        return self.concentration * exp(-x * self.supportive_val)  # drop the term exp(-C)
