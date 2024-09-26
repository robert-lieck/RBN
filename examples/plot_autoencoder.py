"""
Autoencoder
===========

This is an example of defining an autoencoder RBN with a single continuous variable.
"""

# %%
# We use the classes defined in :mod:`rbnet.autoencoder`

import torch

from rbnet.base import SequentialRBN
from rbnet.autoencoder import AutoencoderCell, AutoencoderTransition, AutoencoderNonTermVar, AutoencoderPrior

cell = AutoencoderCell(
    transition=AutoencoderTransition(terminal_encoder=lambda x: x,
                                     terminal_decoder=lambda z, x: torch.ones(1),
                                     # non_terminal_encoder=lambda x1, x2: (x1 + x2) / 2,
                                     non_terminal_encoder=lambda x1, x2: max(x1, x2),
                                     non_terminal_decoder=lambda z, x1, x2: torch.ones(1)),
    variable=AutoencoderNonTermVar(dim=1)
)
rbn = SequentialRBN(cells=[cell], prior=AutoencoderPrior())
print(rbn.inside(torch.linspace(0, 10, 6)[:, None]))
for p in rbn.inside_chart[0].pretty():
    print(p)