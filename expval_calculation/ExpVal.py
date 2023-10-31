from problem_generation.Generating_Problems import Problem


class ExpectationValues:
    """General expectation value calculator; base class."""

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.energy = None
        self.best_energy = None
        self.fixed_correl = []
        self.expect_val_dict = {}
