import numpy as np

class kpoint:
    """A k-point with k -> -k symmetry coordinates and a weight.  Internal use."""

    def __init__(self, k, weight = 1) -> None:
        self.k = np.array(k)
        for i in range(3):
            if self.k[i] > 0.5:
                self.k[i] = self.k[i] -1
            if self.k[i] < -0.5:
                self.k[i] = 1 + self.k[i]
        
        if self.k[-1] < 0:
            self.k = - self.k
        elif self.k[-1] == 0:
            if self.k[-2] < 0:
                self.k = - self.k
            elif self.k[-2] == 0:
                if self.k[-3] < 0:
                    self.k = - self.k

        self.weight = weight

    def __str__(self) -> str:
        strings = ["%.12f" % number for number in self.k + 1e-14] # add zero to avoid -0.0000 in string format
        string = str(strings)
        return string

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

def createGridAndWeights(kpoints=[3, 3, 3], kshift=[0, 0, 0]):
    """Generate a uniform k-point grid with k -> -k symmetry reduction.

    Parameters
    ----------
    kpoints : array-like of int, length 3
        Number of grid points along each reciprocal lattice direction.
    kshift : array-like of int, length 3
        Shift flags: 0 = no shift, 1 = shift by 0.5/N_k in that direction.

    Returns
    -------
    list of tuple
        Each entry is (k_coords, weight) where k_coords is a list of three
        floats (fractional reciprocal coordinates) and weight is a float.
        Weights sum to 1 over all k-points.
    """
    ks = list(kshift)
    kpoints = np.array(kpoints)
    totalPoints = kpoints[0] * kpoints[1] * kpoints[2]
    kgrid = np.zeros((totalPoints, 3))
    weights = np.zeros(totalPoints)
    weights = 1 / totalPoints

    for i in range(3):
        if not (ks[i] == 0 or ks[i] == 1):
            print("Invalid ks value. all shifts must be zero or 1", ks[i])
        if ks[i] == 1:
            ks[i] = 0.5 * 1.0 / kpoints[i]

    ipoint = 0
    ksetDict = {}
    for i in range(kpoints[0]):
        for j in range(kpoints[1]):
            for k in range(kpoints[2]):
                k = np.array([i / kpoints[0] + ks[0], j / kpoints[1] + ks[1], k / kpoints[2] + ks[2]])
                kp = kpoint(k)
                if str(kp) in ksetDict:
                    ksetDict[str(kp)].weight += 1
                else:
                    ksetDict[str(kp)] = kp

    kset = []

    for k in ksetDict.values():
        kset.append(( list(k.k), k.weight / totalPoints ))

    return kset

if __name__ == "__main__":
    kset = createGridAndWeights([3, 2, 1], [0, 1, 0])

    for k in kset:
        print(k)

