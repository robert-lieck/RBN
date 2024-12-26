import torch

from rbnet.base import RBN, Transition
from rbnet.util import ConstrainedModuleList


class SequentialRBN(RBN, torch.nn.Module):

    def __init__(self, cells, prior, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cells = ConstrainedModuleList(cells)
        self._prior = prior
        self.n = None
        self._terminal_chart = None
        self._inside_chart = None
        self._outside_chart = None

    def init_inside(self, sequence):
        self.n = len(sequence)
        self._terminal_chart = sequence
        self._inside_chart = [c.variable.get_chart(self.n) for c in self._cells]

    def inside_schedule(self, *args, **kwargs):
        for span in range(1, self.n + 1):
            for start in range(self.n - span + 1):
                yield start, start + span

    @RBN.root_location.getter
    def root_location(self):
        return 0, self.n

    def cells(self):
        return self._cells

    def update_inside_chart(self, var_idx, locations, values):
        self._inside_chart[var_idx][locations] = values

    @RBN.inside_chart.getter
    def inside_chart(self):
        return self._inside_chart

    @RBN.terminal_chart.getter
    def terminal_chart(self):
        return self._terminal_chart

    @RBN.prior.getter
    def prior(self):
        return self._prior


class SequentialBinaryTransition(Transition):

    def iterate_inside_splits(self, location):
        start, end = location
        if end - start <= 1:
            # no splitting possible
            return
        else:
            for split in range(start + 1, end):
                yield start, split, end


class SequentialTerminalTransition(Transition):

    def iterate_inside_splits(self, location):
        start, end = location
        if end - start > 1:
            # no terminal transition possible
            return
        else:
            yield start
