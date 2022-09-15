from enum import Enum

class ExecutionStrategy(Enum):
    EEBasic = 0
    EEImprovemmentBase = 1
    EEInverseRoullette = 2
    EEMutationCompensation = 3
    EEExplorationCompensation = 4
    EEExplorationCompensationInSchwefel = 5
    EEExplorationCompensationInRastrigin = 6
    GABasic = 7
    GABasicInRastrigin = 8