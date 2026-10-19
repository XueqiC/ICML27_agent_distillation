"""Shared baseline engineering infrastructure."""

REGISTERED_K = (1, 2, 5, 10, 20, 50, 100)


def registered_support(selection):
    """The full-TRAIN cell is identified by its rule, never a magic trusted K."""
    from sed.seed_selection import is_full_train
    return is_full_train(selection) or selection["K"] in REGISTERED_K
