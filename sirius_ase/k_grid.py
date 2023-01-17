import numpy as np

class kpoint:

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
        strings = ["%.10f" % number for number in self.k + 0] # add zero to avoid -0.0000 in string format
        string = str(strings)
        # print(string)
        return string

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, __o: object) -> bool:
        return hash(self) == hash(__o)

def createGridAndWeights(kpoints = [3, 3, 3], kshift = [0, 0, 0]):
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
            print('.asd', ks[i], kpoints[i])

    ipoint = 0
    ksetDict = {}
    for i in range(kpoints[0]):
        for j in range(kpoints[1]):
            for k in range(kpoints[2]):
                k = np.array([i / kpoints[0] + ks[i], j / kpoints[1] + ks[j], k / kpoints[2] + ks[k]])
                kp = kpoint(k)
                if str(kp) in ksetDict:
                    ksetDict[str(kp)].weight += 1
                else:
                    ksetDict[str(kp)] = kp

    kset = []

    for k in ksetDict.values():
        kset.append(( list(k.k), k.weight ))

    return kset

if __name__ == "__main__":
    kset = createGridAndWeights([2, 2, 2], [0, 0, 0])

    for k in kset:
        print(k)

