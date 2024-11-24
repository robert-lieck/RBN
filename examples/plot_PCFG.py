"""
Abstracted PCFGs
================

This is an example of defining a simple discrete RBN equivalent to a PCFG using the :class:`AbstractedPCFG
<rbnet.pcfg.AbstractedPCFG>` class.
"""

from rbnet.pcfg import AbstractedPCFG

# %%
# Minimal Example
# ---------------
# We start with a minimal example (also used in :doc:`/auto_examples/plot_discrete_RBN`):

pcfg = AbstractedPCFG(non_terminals="SAB", terminals="ab", start="S", rules=[
    ("S --> A B", 1), ("S --> B A", 1),  # prior + first transition
    ("A --> B A", 1), ("B --> A B", 1),  # non-terminal transitions
    ("A --> a", 1), ("B --> b", 1)       # terminal transition
])

print(pcfg.inside(sequence="aaaa"))
print(pcfg.inside(sequence="bbbb"))
print(pcfg.inside(sequence="aaab"))
print(pcfg.inside_chart[0].pretty())

# %%
# Defining the PCFG
# -----------------

# %%
# First we define a number of words (terminal symbols) of different categories that sentences can be composed of:
subjects = ["I", "You", "We", "They"]
verbs = ["run", "drink", "sleep"]
adverb_non_gradable = ["a-lot", "alone"]
adverb_gradable = ["fast", "slowly", "quickly"]
grade = ["very", "veeery", "really"]
verb_qualifier = ["rarely", "do-not", "never", "always"]
terminals = subjects + verbs + adverb_non_gradable + adverb_gradable + grade + verb_qualifier

# %%
# Then we define some non-terminal symbols (a start symbol and one symbol for each category of words used above):
non_terminals = ["start",
                 "subject",
                 "verb",
                 "gradable_adverb",
                 "non_gradable_adverb",
                 "verb_qualifier",
                 "grade"]

# %%
# Finally, we define the rules and give them a weight (for simplicity we use a weight of 1 everywhere):
non_terminal_rules = [("start --> subject verb", 1),
                      ("verb --> verb_qualifier verb", 1),
                      ("verb --> verb gradable_adverb", 1),
                      ("verb --> verb non_gradable_adverb", 1),
                      ("gradable_adverb --> grade gradable_adverb", 1),
                      ("grade --> grade grade", 1)]
terminal_rules = []
for non_terminal_symbol, corresponding_list_of_terminal_symbols in zip(
        non_terminals[1:],  # skip the start symbol
        [subjects, verbs, adverb_gradable, adverb_non_gradable, verb_qualifier, grade]
):
    for terminal_symbol in corresponding_list_of_terminal_symbols:
        terminal_rules.append((f"{non_terminal_symbol} --> {terminal_symbol}", 1))

# %%
# Now we can define our PCFG by providing it with the terminals, non-terminals, rules, and start symbol.
pcfg = AbstractedPCFG(terminals=terminals,
                      non_terminals=non_terminals,
                      rules=non_terminal_rules + terminal_rules,
                      start="start")
# %%
# Parsing Sentences
# -----------------
# Let's test the grammar by computing the marginal likelihood of some grammatical sentences
# (which should be greater than zero) and for some un-grammatical ones (which should have zero marginal likelihood)

grammatical_sentences = [
    "I run",
    "You never run",
    "We run very veeery slowly",
    "They always run alone",
    "I never sleep really very quickly",
    "You do-not drink very quickly"]
ungrammatical_sentences = [
    "I You",
    "run fast"
]
for sentence in grammatical_sentences + ungrammatical_sentences:
    marginal_likelihood = pcfg.inside(sequence=sentence.split())
    print(f"{sentence} --> {marginal_likelihood}")

# %%
# We can also print a simple textual visualisation of the parse chart, which shows
# ``non-terminal symbol|inside probability`` at each location
pcfg.inside(sequence="You never run".split())
print(pcfg.map_inside_chart(precision=2).pretty())

# %%
# Training Parameters
# -------------------
# For a given dataset of sentences, we can train the model parameters
import pytorch_lightning as pl
import torch
import numpy as np
from rbnet.util import SequenceDataModule

print(pcfg)
pcfg.auto_tokenise = False
# pcfg.cells[0].variable.chart_type = "dict"
data = SequenceDataModule([pcfg.tokenise(s.split()) for s in grammatical_sentences], val_split=0, test_split=0)
data.setup()

# for batch in data.train_dataloader():
#     print(batch[0])
#     print(pcfg.inside(batch[0]))
#
# for s in grammatical_sentences:
#     s = pcfg.tokenise(s.split())
#     print(s)
#     print(pcfg.inside(s))

# print(list(pcfg.parameters()))



# trainer = pl.Trainer(max_epochs=100)
# trainer.fit(pcfg, data.train_dataloader())
#
# print(list(pcfg.parameters()))
