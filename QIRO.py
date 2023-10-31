class QIRO:
    """General QIRO base class"""

    def __init__(self, nc, expectation_values):
        self.problem = expectation_values.problem
        self.nc = nc
        self.expectation_values = expectation_values
        self.assignment = []
        self.solution = []
