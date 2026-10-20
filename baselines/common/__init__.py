"""Shared baseline engineering infrastructure."""

REGISTERED_K = (1, 2, 5, 10, 20, 50, 100)


def registered_support(selection):
    """The full-TRAIN cell is identified by its rule, never a magic trusted K."""
    from sed.seed_selection import is_full_train, is_stratified_nested
    return is_full_train(selection) or is_stratified_nested(selection) or selection["K"] in REGISTERED_K
