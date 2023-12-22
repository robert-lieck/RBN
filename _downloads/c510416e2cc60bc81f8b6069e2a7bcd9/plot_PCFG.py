"""
Abstracted PCFGs
================

This is an example of defining a simple discrete RBN equivalent to a PCFG.
"""

# %%
# Defining the PCFG
# -----------------
# We use an :class:`AbstractedPCFG <rbnet.pcfg.AbstractedPCFG>`
from rbnet.pcfg import AbstractedPCFG

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
# Then we define some non-terminals symbols (a start symbol and one symbol for each category of words used above):
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
for sentence in [
    # grammatical
    "I run",
    "You never run",
    "We run very veeery slowly",
    "They always run alone",
    "I never sleep really very quickly",
    "You do-not drink very quickly",
    # un-grammatical
    "I You",
    "run fast"
]:
    marginal_likelihood = pcfg.inside(sequence=sentence.split())
    print(f"{sentence} --> {marginal_likelihood}")

# %%
# We can also print a simple textual visualisation of the parse chart, which shows
# ``non-terminal symbol|inside probability`` at each location
pcfg.inside(sequence="You never run".split())
print(pcfg.map_inside_chart(precision=2).pretty())
