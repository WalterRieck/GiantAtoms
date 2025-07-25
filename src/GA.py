import numpy as np
from math import comb
from joblib import dump, load


class GA:
    def __init__(self, N, J, nAtoms, gs, deltas, couplePoints):
        # gs, deltas and couplePoints are arrays with length nAtoms, the elements in couplePoints are arrays with a
        # number of couple-points, N is the number of cavities in the bath

        self.N = N
        self.nAtoms = nAtoms
        self.gs = gs
        self.deltas = deltas
        self.couplePoints = couplePoints
        self.J = J
        self.Hamiltonian = None
        self.es = None
        self.vs = None
        self.finalStates = None

    def ConstructHamiltonian(self):
        H = np.zeros((self.nAtoms + self.N, self.nAtoms + self.N))
        for i in range(self.nAtoms):
            delta = self.deltas[i]
            couplePointAtom = self.couplePoints[i]
            g = self.gs[i]
            iter = 0
            for couplePoint in couplePointAtom:
                H[i, int(couplePoint) + self.nAtoms] = g[iter]
                H[int(couplePoint) + self.nAtoms, i] = g[iter]
                iter += 1
            H[i, i] = delta

        for i in range(self.nAtoms, self.nAtoms + self.N):
            for j in range(i, self.nAtoms + self.N):
                if np.abs(i - j) == 1:
                    H[i, j] = -self.J
                    H[j, i] = -self.J

        self.Hamiltonian = H

    def computeEigens(self):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        self.es, self.vs = np.linalg.eigh(self.Hamiltonian)
        return self.es, self.vs

    def constructInitialState(self, atom, cavity):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()

        initialState = np.zeros(len(self.Hamiltonian))
        if atom == 0:
            initialState[cavity - 1 + self.nAtoms] = 1
        else:
            initialState[atom - 1] = 1
        return initialState

    def computeDynamics(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns = np.zeros((self.nAtoms, len(ts)))

        phases = np.exp(-1j * np.outer(ts, self.es))
        projections = np.dot(self.vs.conj().T, initialState)
        finalStates = (self.vs @ (phases * projections).T).T

        for i in range(self.nAtoms):
            iter = 0
            ns[i] = np.abs(finalStates[:, i]) ** 2
        self.finalStates = finalStates
        self.dynamics = ns
        return self.dynamics
    
    def computeSiteDynamics(self, ts, initialState):
        if self.finalStates is None:
            self.computeDynamics(ts, initialState)
        nt = len(ts)
        N = self.N

        result_n = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            for k in range(self.N):
                result_n[i, k] = np.sum(np.abs(self.finalStates[i, k + self.nAtoms]) ** 2)

        return result_n
        
    def computeWavepacketDynamics(self, ts, initialState):   # how the wavepacket evolves in the waveguide
        if self.es is None or self.vs is None:
            self.computeEigens()
        
        nt=len(ts)
        N=self.N

        result_n=np.zeros((nt, N))
        
        for i in range(nt):
            t=ts[i]
            finalState = np.zeros(self.N + self.nAtoms, dtype=complex)
            for j in range(self.N + self.nAtoms):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]), initialState)
            
            for k in range(self.N):
                result_n[i, k]=np.abs(finalState[k+self.nAtoms]) ** 2
                
        return result_n
        

    def computeIPR(self):
        if self.vs is None:
            self.computeEigens()

        IPR = np.zeros(self.N + self.nAtoms)
        for i in range(self.N + self.nAtoms):
            v = self.vs[:, i]
            for j in range(self.N + self.nAtoms):
                IPR[i] += np.abs(v[j]) ** 4

        self.IPR = IPR
        return IPR

    def saveGA(self, filename):
        dump(self, filename)

    @classmethod
    def loadGA(cls, filename):
        return load(filename)


class SA:
    def __init__(self, N, J, nAtoms, gs, deltas, couplePoints):
        # gs, deltas and couplePoints are arrays with length nAtoms, N is the number of cavities in the bath

        self.N = N
        self.nAtoms = nAtoms
        self.gs = gs
        self.deltas = deltas
        self.couplePoints = couplePoints
        self.J = J
        self.Hamiltonian = None
        self.es = None
        self.vs = None

    def ConstructHamiltonian(self):
        H = np.zeros((self.nAtoms + self.N, self.nAtoms + self.N))
        for i in range(self.nAtoms):
            delta = self.deltas[i]
            couplePoint = self.couplePoints[i]
            g = self.gs[i]
            H[i, int(couplePoint) + self.nAtoms] = g
            H[int(couplePoint) + self.nAtoms, i] = g
            H[i, i] = delta

        for i in range(self.nAtoms, self.nAtoms + self.N):
            for j in range(i, self.nAtoms + self.N):
                if np.abs(i - j) == 1:
                    H[i, j] = -self.J
                    H[j, i] = -self.J

        self.Hamiltonian = H

    def computeEigens(self):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        self.es, self.vs = np.linalg.eigh(self.Hamiltonian)
        return self.es, self.vs

    def constructInitialState(self, atom, cavity):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        initialState = np.zeros(len(self.Hamiltonian))
        if atom == 0:
            initialState[cavity - 1 + self.nAtoms] = 1
        else:
            initialState[atom - 1] = 1
        return initialState

    def computeDynamics(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns = np.zeros((self.nAtoms, len(ts)))
        for i in range(self.nAtoms):
            iter = 0
            for t in ts:
                finalState = np.zeros(self.N + self.nAtoms, dtype=complex)
                for j in range(self.N + self.nAtoms):
                    finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]), initialState)
                ns[i, iter] = np.abs(finalState[i]) ** 2
                iter += 1

        self.dynamics = ns
        return self.dynamics

    def computeWavepacketDynamics(self, ts, initialState):  # how the wavepacket evolves in the waveguide
        if self.es is None or self.vs is None:
            self.computeEigens()

        nt = len(ts)
        N = self.N

        result_n = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            finalState = np.zeros(self.N + self.nAtoms, dtype=complex)
            for j in range(self.N + self.nAtoms):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                    initialState)

            for k in range(self.N):
                result_n[i, k] = np.abs(finalState[k + 1]) ** 2

        return result_n

    def computeIPR(self):
        if self.vs is None:
            self.computeEigens()

        IPR = np.zeros(self.N + self.nAtoms)
        for i in range(self.N + self.nAtoms):
            v = self.vs[:, i]
            for j in range(self.N + self.nAtoms):
                IPR[i] += np.abs(v[j]) ** 4

        self.IPR = IPR
        return IPR

    def saveGA(self, filename):
        dump(self, filename)

    @classmethod
    def loadGA(cls, filename):
        return load(filename)


class TSA:
    def __init__(self, N, J, nAtoms, gs, deltas2, deltas1, couplePoints):
        self.N = N
        self.nAtoms = nAtoms
        self.gs = gs
        self.deltas2 = deltas2
        self.deltas1 = deltas1
        self.couplePoints = couplePoints
        self.J = J
        self.Hamiltonian = None
        self.es = None
        self.vs = None

    def ConstructHamiltonian(self):
        H = np.zeros((comb(self.N + self.nAtoms + 1, 2), comb(self.N + self.nAtoms + 1, 2)))
        for i in range(self.nAtoms):
            for j in range(len(H)):
                if j < self.nAtoms:
                    H[j, j] = self.deltas2[i] + self.deltas1[i]
                    H[j, int(self.couplePoints[i]) + i * self.N + comb(self.nAtoms+1, 2) - 1] = np.sqrt(2) * self.gs[i]
                    H[int(self.couplePoints[i]) + i * self.N + comb(self.nAtoms+1, 2) - 1, j] = np.sqrt(2) * self.gs[i]
                elif j < comb(self.nAtoms + 1, 2):
                    H[j, j] = self.deltas1[i]
                    H[j, int(self.couplePoints[i]) + i * self.N + comb(self.nAtoms + 1, 2)] = self.gs[i]
                    H[int(self.couplePoints[i]) + i * self.N + comb(self.nAtoms + 1, 2), j] = self.gs[i]
                elif j < comb(self.nAtoms + 1, 2) + self.nAtoms * self.N:
                    H[j, j] = self.deltas1[i]
                    if j != comb(self.nAtoms + 1, 2):
                        H[j, j - 1] = -self.J
                        H[j - 1, j] = -self.J
                    ii = j - comb(self.nAtoms + 1, 2) - i * self.N
                    jj1 = ii
                    kk1 = self.couplePoints[i] - 1
                    if int(jj1) <= int(kk1):
                        jk1 = jj1*self.N - jj1*(jj1 - 1)/2 + kk1 - jj1
                        H[j, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = self.gs[i]
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), j] = self.gs[i]
                    jj2 = self.couplePoints[i] - 1
                    kk2 = ii
                    if int(jj2) <= int(kk2):
                        jk2 = jj2*self.N - jj2*(jj2 - 1)/2 + kk2 - jj2
                        H[j, int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = self.gs[i]
                        H[int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), j] = self.gs[i]

                else:
                    jk = j - (comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)
                    jj = 0
                    kk = 0
                    s = "fail"
                    for k in range(0, self.N):
                        if jk < k * self.N - k*(k-1) / 2:
                            kk = jk - (k - 1)*self.N + (k - 2)*(k - 1) / 2 + (k - 1)
                            jj = k - 1
                            break
                        jj = self.N - 1
                        kk = self.N - 1
                    if int(jj) == 0 and int(kk) == 0:
                        jj1 = jj + 1
                        kk1 = kk
                        jj2 = jj
                        kk2 = kk + 1
                    elif int(jj) == 0 or int(kk) == self.N - 1:
                        jj1 = jj + 1
                        kk1 = kk
                        jj2 = jj
                        kk2 = kk - 1
                    elif int(jj) == self.N - 1 or int(kk) == 0:
                        jj1 = jj - 1
                        kk1 = kk
                        jj2 = jj
                        kk2 = kk + 1
                    elif int(jj) == self.N - 1 and int(kk) == self.N - 1:
                        jj1 = jj - 1
                        kk1 = kk
                        jj2 = jj
                        kk2 = kk - 1
                    else:
                        jj1 = jj - 1
                        jj2 = jj + 1
                        jj3 = jj
                        jj4 = jj
                        kk1 = kk
                        kk2 = kk
                        kk3 = kk + 1
                        kk4 = kk - 1
                        s = "pass"

                    if int(jj1) <= int(kk1):
                        jk1 = jj1 * self.N - jj1*(jj1 - 1)/2 + kk1 - jj1
                        H[j, int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), j] = -self.J
                    if int(jj2) <= int(kk2):
                        jk2 = jj2 * self.N - jj2*(jj2 - 1)/2 + kk2 - jj2
                        H[j, int(jk2 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                        H[int(jk2 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), j] = -self.J
                    if s == "pass":
                        if int(jj3) <= int(kk3):
                            jk3 = jj3 * self.N - jj3 * (jj3 - 1) / 2 + kk3 - jj3
                            H[j, int(jk3 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                            H[int(jk3 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), j] = -self.J
                        if int(jj4) <= int(kk4):
                            jk4 = jj4 * self.N - jj4 * (jj4 - 1) / 2 + kk4 - jj4
                            H[j, int(jk4 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                            H[int(jk4 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), j] = -self.J

        self.Hamiltonian = H

    def computeEigens(self):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        self.es, self.vs = np.linalg.eigh(self.Hamiltonian)
        return self.es, self.vs

    def constructInitialState(self, atoms, cavities):  # IMPORTANT THAT ATOMS[0] < ATOMS[1] AND CAVITIES[0] <= CAVITIES[1]
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        initState = np.zeros(len(self.Hamiltonian))
        if sum(atoms) == 0:
            initState[comb(self.nAtoms + 1, 2) + self.nAtoms * self.N - 1 + cavities[0] * self.N - cavities[0]*(cavities[0] - 1)/2 + cavities[1] - cavities[0]] = 1
        elif sum(cavities) == 0:
            if atoms[0] == 0 or atoms[1] == 0:
                initState[sum(atoms) - 1] = 1
            else:
                initState[atoms[0] * self.N - atoms[0]*(atoms[0] - 1)/2 + atoms[1] - atoms[0]] = 0
        else:
            initState[comb(self.nAtoms + 1, 2) + sum(cavities) + (sum(atoms) - 1) * self.N - 1] = 1
        return initState

    def computeDynamics(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        for i in range(self.nAtoms):
            iter = 0
            for t in ts:
                finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
                for j in range(len(self.Hamiltonian)):
                    finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                        initialState)
                ns2[i, iter] = np.abs(finalState[i]) ** 2
                ns1[i, iter] = np.sum(np.abs(finalState[self.nAtoms:self.nAtoms + self.N]) ** 2)
                iter += 1

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        return self.dynamicsl2, self.dynamicsl1

    def computeDynamics2_0(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        EE = np.zeros(len(ts))

        phases = np.exp(-1j * np.outer(ts, self.es))
        projections = np.dot(self.vs.conj().T, initialState)
        finalStates = (self.vs @ (phases * projections).T).T

        indexes = []
        for i in range(self.nAtoms):
            for j in range(self.nAtoms):
                if i == j:
                    continue
                elif i < j:
                    indexes.append(int(self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1))
                else:
                    indexes.append(int(self.nAtoms + j * self.nAtoms - (j - 1) / 2 + i - 2 * j - 1))

            ns2[i] = np.abs(finalStates[:, i]) ** 2
            ns1[i] = (np.sum(np.abs(finalStates[:, comb(self.nAtoms + 1, 2) + self.N * i:comb(self.nAtoms + 1, 2) + self.N * (i + 1)]) ** 2,axis=1)
                    + np.sum(np.abs(finalStates[:, indexes]) ** 2, axis=1))
        EE = np.abs(finalStates[:, 2]) ** 2

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        self.dynamicsEE = EE
        self.finalStates = finalStates

        return self.dynamicsl2, self.dynamicsl1, EE

    def computeSiteDynamics(self, ts, initialState, index):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns = np.zeros((1, len(ts)))
        for i in range(1):
            iter = 0
            for t in ts:
                finalState = np.zeros(self.N + self.nAtoms, dtype=complex)
                for j in range(self.N + self.nAtoms):
                    finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                        initialState)
                ns[i, iter] = np.abs(finalState[index]) ** 2
                iter += 1

        self.dynamics = ns
        return self.dynamics

    def computeWavepacketDynamics(self, ts, initialState):  # how the wavepackt evolves in the waveguide
        if self.es is None or self.vs is None:
            self.computeEigens()

        nt = len(ts)
        N = self.N

        result_n = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
            for j in range(len(self.Hamiltonian)):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                    initialState)
            for k in range(self.N):
                indeces = []
                for l in range(self.nAtoms):
                    indeces.append(int(comb(self.nAtoms + 1, 2) + l * self.N + k))
                for l in range(self.N):
                    if l >= k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))
                    else:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + l * self.N - l * (l - 1) / 2 + k - l))
                result_n[i, k] = np.sum(np.abs(finalState[indeces]) ** 2)

        return result_n

    def computeIPR(self):
        if self.vs is None:
            self.computeEigens()

        IPR = np.zeros(self.N + self.nAtoms)
        for i in range(self.N + self.nAtoms):
            v = self.vs[:, i]
            for j in range(self.N + self.nAtoms):
                IPR[i] += np.abs(v[j]) ** 4

        self.IPR = IPR
        return IPR

    def saveGA(self, filename):
        dump(self, filename)

    @classmethod
    def loadGA(cls, filename):
        return load(filename)


class TGA:
    def __init__(self, N, J, nAtoms, gs, deltas2, deltas1, couplePoints):
        self.N = N
        self.nAtoms = nAtoms
        self.gs = gs
        self.deltas2 = deltas2
        self.deltas1 = deltas1
        self.couplePoints = couplePoints
        self.J = J
        self.Hamiltonian = None
        self.es = None
        self.vs = None
        self.finalStates = None

    def ConstructHamiltonian(self):
        H = np.zeros((comb(self.N + self.nAtoms + 1, 2), comb(self.N + self.nAtoms + 1, 2)))
        for i in range(comb(self.nAtoms + 1, 2)):
            if i < self.nAtoms:
                couplePoints = self.couplePoints[i]
                H[i, i] = self.deltas2[i] + self.deltas1[i]
                iter = 0
                for couplePoint in couplePoints:
                    H[i, comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1] = np.sqrt(2) * self.gs[i][iter]
                    H[comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1, i] = np.sqrt(2) * self.gs[i][iter]
                    iter += 1
            else:
                H[i, i] = sum(self.deltas1)
        for i in range(self.nAtoms - 1):
            for j in range(i + 1, self.nAtoms):
                couplePoints1 = self.couplePoints[i]
                couplePoints2 = self.couplePoints[j]
                index = self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1
                iter = 0
                for couplePoint in couplePoints1:
                    H[int(index), comb(self.nAtoms + 1, 2) + j * self.N + couplePoint - 1] = np.sqrt(1) * self.gs[i][
                        iter]
                    H[comb(self.nAtoms + 1, 2) + j * self.N + couplePoint - 1, int(index)] = np.sqrt(1) * self.gs[i][
                        iter]
                    iter += 1
                iter = 0
                for couplePoint in couplePoints2:
                    H[int(index), comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1] = np.sqrt(1) * self.gs[j][
                        iter]
                    H[comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1, int(index)] = np.sqrt(1) * self.gs[j][
                        iter]
                    iter += 1

        for i in range(self.nAtoms):
            for j in range(self.N):
                index = self.N * i + j + comb(self.nAtoms + 1, 2)
                H[index, index] = self.deltas1[i]
                if j != self.N - 1:
                    H[index, index + 1] = -self.J
                    H[index + 1, index] = -self.J
                couplePoints = self.couplePoints[i]
                iter = 0
                for couplePoint in couplePoints:
                    jj1 = j
                    kk1 = couplePoint - 1
                    if int(jj1) < int(kk1):
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(1) * self.gs[i][
                            iter]
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(1) * self.gs[i][
                            iter]
                    elif int(jj1) == int(kk1):
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(2) * self.gs[i][
                            iter]
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(2) * self.gs[i][
                            iter]

                    jj2 = couplePoint - 1
                    kk2 = j
                    if int(jj2) < int(kk2):
                        jk2 = jj2 * self.N - jj2 * (jj2 - 1) / 2 + kk2 - jj2
                        H[index, int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(1) * self.gs[i][
                            iter]
                        H[int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(1) * self.gs[i][
                            iter]
                    elif int(jj2) == int(kk2):
                        jk2 = jj2 * self.N - jj2 * (jj2 - 1) / 2 + kk2 - jj2
                        H[index, int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(2) * self.gs[i][
                            iter]
                        H[int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(2) * self.gs[i][
                            iter]
                    iter += 1


        for i in range(comb(self.N + 1, 2)):
            index = i + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N
            jk = i
            jj = 0
            kk = 0
            for k in range(0, self.N):
                if jk < k * self.N - k * (k - 1) / 2:
                    kk = int(jk - (k - 1) * self.N + (k - 2) * (k - 1) / 2 + (k - 1))
                    jj = int(k - 1)
                    break
                jj = int(self.N - 1)
                kk = int(self.N - 1)

            jjkks = [[int(jj + 1), kk], [jj, int(kk + 1)], [int(jj - 1), kk], [jj, int(kk - 1)]]

            for [jj1, kk1] in jjkks:
                if jj1 < 0  or jj1 > self.N - 1:
                    continue
                elif kk1 < 0 or kk1 > self.N - 1:
                    continue
                elif jj1 > kk1:
                    continue
                else:
                    if jj1 == kk1 or jj == kk:
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -np.sqrt(2) * self.J
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), index] = -np.sqrt(2) * self.J
                    else:
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), index] = -self.J

        self.Hamiltonian = H

    def computeEigens(self):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        self.es, self.vs = np.linalg.eigh(self.Hamiltonian)
        return self.es, self.vs

    def constructInitialState(self, atoms,
                              cavities):  # IMPORTANT THAT ATOMS[0] < ATOMS[1] AND CAVITIES[0] <= CAVITIES[1]
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        initState = np.zeros(len(self.Hamiltonian))
        if sum(atoms) == 0:
            initState[int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N - 1 + cavities[0] * self.N - cavities[0] * (
                    cavities[0] - 1) / 2 + cavities[1] - cavities[0])] = 1
        elif sum(cavities) == 0:
            if atoms[0] == 0 or atoms[1] == 0:
                initState[sum(atoms) - 1] = 1
            else:
                initState[int(self.nAtoms - atoms[0] * (atoms[0] - 1) / 2 + atoms[1] - atoms[0] - 1)] = 1
        else:
            initState[comb(self.nAtoms + 1, 2) + sum(cavities) + (sum(atoms) - 1) * self.N - 1] = 1
        return initState

    def computeDynamics2_0(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        EE = np.zeros(len(ts))

        phases = np.exp(-1j * np.outer(ts, self.es))
        projections = np.dot(self.vs.conj().T, initialState)
        finalStates = (self.vs @ (phases * projections).T).T

        indexes = []
        for i in range(self.nAtoms):
            for j in range(self.nAtoms):
                if i == j:
                    continue
                elif i < j:
                    indexes.append(int(self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1))
                else:
                    indexes.append(int(self.nAtoms + j * self.nAtoms - (j - 1) / 2 + i - 2 * j - 1))

            ns2[i] = np.abs(finalStates[:, i]) ** 2
            ns1[i] = (np.sum(np.abs(finalStates[:, comb(self.nAtoms + 1, 2) + self.N * i:comb(self.nAtoms + 1, 2) + self.N * (i + 1)]) ** 2,axis=1)
                    + np.sum(np.abs(finalStates[:, indexes]) ** 2, axis=1))
        EE = np.abs(finalStates[:, 2]) ** 2

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        self.dynamicsEE = EE
        self.finalStates = finalStates

        return self.dynamicsl2, self.dynamicsl1, EE

    def computeDynamics(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        EE = np.zeros(len(ts))
        indexes = []
        iter = 0
        for t in ts:
            finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
            for j in range(len(self.Hamiltonian)):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]), initialState)
            for i in range(self.nAtoms):
                for j in range(self.nAtoms):
                    if i == j:
                        continue
                    elif i < j:
                        indexes.append(int(self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1))
                    else:
                        indexes.append(int(self.nAtoms + j * self.nAtoms - (j - 1) / 2 + i - 2 * j - 1))
                EE[iter] = np.abs(finalState[2]) ** 2
                ns2[i, iter] = np.abs(finalState[i]) ** 2
                ns1[i, iter] = (np.sum(np.abs(finalState[(comb(self.nAtoms + 1, 2) + self.N* i):(comb(self.nAtoms + 1, 2) + self.N * (i + 1))]) ** 2)
                              + np.sum(np.abs(finalState[indexes]) ** 2))

            iter += 1
            if iter % 100 == 0:
                print(iter)

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        self.dynamicsEE = EE
        return self.dynamicsl2, self.dynamicsl1, EE

    def computeSiteDynamics(self, ts, initialState):
        if self.finalStates is None:
            self.computeDynamics2_0(ts, initialState)
        nt = len(ts)
        N = self.N

        resultE = np.zeros((nt, N))
        resultG = np.zeros((nt, N))
        resultGG = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            for k in range(self.N):
                indeces = []
                for l in range(self.nAtoms):
                    indeces.append(int(comb(self.nAtoms + 1, 2) + l * self.N + k))
                resultE[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)
                indeces = []
                for l in range(self.N):
                    if l >= k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))
                    else:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + l * self.N - l * (l - 1) / 2 + k - l))
                resultG[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)
                indeces = []
                for l in range(self.N):
                    if l == k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))

                    resultGG[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)

        return resultE, resultG, resultGG

    def computeWavepacketDynamics(self, ts, initialState):  # how the wavepackt evolves in the waveguide
        if self.es is None or self.vs is None:
            self.computeEigens()

        nt = len(ts)
        N = self.N

        result_n = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
            for j in range(len(self.Hamiltonian)):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                    initialState)

            for k in range(self.N):
                indeces = []
                for l in range(self.nAtoms):
                    indeces.append(int(comb(self.nAtoms + 1, 2) + l * self.N + k))
                for l in range(self.N):
                    if l >= k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))
                    else:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + l * self.N - l * (l - 1) / 2 + k - l))
                result_n[i, k] = np.sum(np.abs(finalState[indeces]) ** 2)

        return result_n

    def computeIPR(self):
        if self.vs is None:
            self.computeEigens()

        IPR = np.zeros(len(self.Hamiltonian))
        for i in range(len(self.Hamiltonian)):
            v = self.vs[:, i]
            for j in range(len(self.Hamiltonian)):
                IPR[i] += np.abs(v[j]) ** 4

        self.IPR = IPR
        return IPR

    def lookUpState(self, index):   # Index ---> atom and cavity numbers, index starts with 0, atoms/cavities with 1
        atoms = []
        cavities = []
        tempIndex = 0
        if index < self.nAtoms:
            atoms.append(index + 1)
            return atoms, cavities
        elif index < self.nAtoms + comb(self.nAtoms, 2):
            tempIndex = index - self.nAtoms
            for i in range(self.nAtoms):
                for j in range(i + 1, self.nAtoms):
                    atomIndex = i * self.nAtoms - i * (i - 1) / 2 + j - 2 * i - 1
                    if tempIndex == atomIndex:
                        atoms.append(i + 1)
                        atoms.append(j + 1)
                        return atoms, cavities
        elif index < self.nAtoms + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms:
            tempIndex = index - self.nAtoms - comb(self.nAtoms, 2)
            q, mod = divmod(tempIndex, self.N)
            atoms.append(q + 1)
            cavities.append(mod + 1)
            return atoms, cavities
        else:
            tempIndex = index - self.nAtoms - comb(self.nAtoms + 1, 2) - self.N * self.nAtoms
            for i in range(self.N):
                for j in range(i, self.N):
                    cavityIndex = i * self.N - i * (i - 1) / 2 + j - i
                    if tempIndex == cavityIndex:
                        cavities.append(i + 1)
                        cavities.append(j + 1)
                        return atoms, cavities

    def saveGA(self, filename):
        dump(self, filename)

    @classmethod
    def loadGA(cls, filename):
        return load(filename)


class TGADoublon:
    def __init__(self, N, J, U, nAtoms, gDC, gs2, gs1, deltas2, deltas1, couplePoints, ):
        self.dynamicsEE = None
        self.dynamicsl1 = None
        self.dynamicsl2 = None
        self.N = N
        self.U = U
        self.gDC = gDC
        self.nAtoms = nAtoms
        self.gs2 = gs2
        self.gs1 = gs1
        self.deltas2 = deltas2
        self.deltas1 = deltas1
        self.couplePoints = couplePoints
        self.J = J
        self.resultG = None
        self.resultGG = None
        self.resultE = None
        self.Hamiltonian = None
        self.es = None
        self.vs = None
        self.finalStates = None
        self.IPR = None

    def ConstructHamiltonian(self):
        H = np.zeros((comb(self.N + self.nAtoms + 1, 2), comb(self.N + self.nAtoms + 1, 2)))
        for i in range(comb(self.nAtoms + 1, 2)):
            if i < self.nAtoms:
                couplePoints = self.couplePoints[i]
                H[i, i] = self.deltas2[i] + self.deltas1[i]
                iter = 0
                for couplePoint in couplePoints:
                    H[i, comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1] = np.sqrt(2) * self.gs2[i][iter]
                    H[comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1, i] = np.sqrt(2) * self.gs2[i][iter]
                    iter += 1
            else:
                H[i, i] = sum(self.deltas1)
        for i in range(self.nAtoms - 1):
            for j in range(i + 1, self.nAtoms):
                couplePoints1 = self.couplePoints[i]
                couplePoints2 = self.couplePoints[j]
                index = self.nAtoms + i * self.nAtoms - i*(i - 1)/2 + j - 2 * i - 1
                iter = 0
                for couplePoint in couplePoints1:
                    H[int(index), comb(self.nAtoms + 1, 2) + j * self.N + couplePoint - 1] = 1*self.gs2[i][iter]
                    H[comb(self.nAtoms + 1, 2) + j * self.N + couplePoint - 1, int(index)] = 1*self.gs2[i][iter]
                    iter += 1
                iter = 0
                for couplePoint in couplePoints2:
                    H[int(index), comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1] = 1*self.gs2[j][iter]
                    H[comb(self.nAtoms + 1, 2) + i * self.N + couplePoint - 1, int(index)] = 1*self.gs2[j][iter]
                    iter += 1

        for i in range(self.nAtoms):
            for j in range(self.N):
                index = self.N * i + j + comb(self.nAtoms + 1, 2)
                H[index, index] = self.deltas1[i]
                if j != self.N - 1:
                    H[index, index + 1] = -self.J
                    H[index + 1, index] = -self.J
                couplePoints = self.couplePoints[i]
                iter = 0
                for couplePoint in couplePoints:
                    jj1 = j
                    kk1 = couplePoint - 1
                    if int(jj1) <= int(kk1):
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = 1 * self.gs1[i][iter]
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = 1 * self.gs1[i][iter]
                        if int(jj1) == int(kk1):
                            H[i, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = 1 * self.gDC[i][iter]
                            H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), i] = 1 * self.gDC[i][iter]
                            H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(2) * self.gs1[i][iter]
                            H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(2) * self.gs1[i][iter]
                    jj2 = couplePoint - 1
                    kk2 = j
                    if int(jj2) <= int(kk2):
                        jk2 = jj2 * self.N - jj2 * (jj2 - 1) / 2 + kk2 - jj2
                        H[index, int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = 1 * self.gs1[i][iter]
                        H[int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = 1 * self.gs1[i][iter]
                        if int(jj2) == int(kk2):
                            H[i, int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = 1 * self.gDC[i][iter]
                            H[int(jk1 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), i] = 1 * self.gDC[i][iter]
                            H[index, int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms)] = np.sqrt(2) * self.gs1[i][iter]
                            H[int(jk2 + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms), index] = np.sqrt(2) * self.gs1[i][iter]
                    iter += 1

        for i in range(comb(self.N + 1, 2)):
            index = i + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N
            jk = i
            jj = 0
            kk = 0
            for k in range(0, self.N):
                if jk < k * self.N - k * (k - 1) / 2:
                    kk = int(jk - (k - 1) * self.N + (k - 2) * (k - 1) / 2 + (k - 1))
                    jj = int(k - 1)
                    break
                jj = int(self.N - 1)
                kk = int(self.N - 1)

            jjkks = [[int(jj + 1), kk], [jj, int(kk + 1)], [int(jj - 1), kk], [jj, int(kk - 1)]]

            for [jj1, kk1] in jjkks:
                if jj1 < 0 or jj1 > self.N - 1:
                    continue
                elif kk1 < 0 or kk1 > self.N - 1:
                    continue
                elif jj1 > kk1:
                    continue
                else:
                    if jj1 == kk1 or jj == kk:
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -np.sqrt(2) * self.J
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), index] = -np.sqrt(2) * self.J
                        if jj1 == kk1:
                            H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = self.U
                    else:
                        jk1 = jj1 * self.N - jj1 * (jj1 - 1) / 2 + kk1 - jj1
                        H[index, int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N)] = -self.J
                        H[int(jk1 + comb(self.nAtoms + 1, 2) + self.nAtoms * self.N), index] = -self.J

        self.Hamiltonian = H

    def computeEigens(self):
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        self.es, self.vs = np.linalg.eigh(self.Hamiltonian)
        return self.es, self.vs

    def constructInitialState(self, atoms,
                              cavities):  # IMPORTANT THAT ATOMS[0] < ATOMS[1] AND CAVITIES[0] <= CAVITIES[1]
        if self.Hamiltonian is None:
            self.ConstructHamiltonian()
        initState = np.zeros(len(self.Hamiltonian))
        if sum(atoms) == 0:
            initState[int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N - 1 + cavities[0] * self.N - cavities[0] * (
                    cavities[0] - 1) / 2 + cavities[1] - cavities[0])] = 1
        elif sum(cavities) == 0:
            if atoms[0] == 0 or atoms[1] == 0:
                initState[sum(atoms) - 1] = 1
            else:
                initState[int(self.nAtoms - atoms[0] * (atoms[0] - 1) / 2 + atoms[1] - atoms[0] - 1)] = 1
        else:
            initState[comb(self.nAtoms + 1, 2) + sum(cavities) + (sum(atoms) - 1) * self.N - 1] = 1
        return initState

    def computeDynamics2_0(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        EE = np.zeros(len(ts))

        phases = np.exp(-1j * np.outer(ts, self.es))
        projections = np.dot(self.vs.conj().T, initialState)
        finalStates = (self.vs @ (phases * projections).T).T

        for i in range(self.nAtoms):
            indexes = []
            for j in range(self.nAtoms):
                if i == j:
                    continue
                elif i < j:
                    indexes.append(int(self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1))
                else:
                    indexes.append(int(self.nAtoms + j * self.nAtoms - (j - 1) / 2 + i - 2 * j - 1))
            ns2[i] = np.abs(finalStates[:, i]) ** 2
            ns1[i] = (np.sum(np.abs(finalStates[:, comb(self.nAtoms + 1, 2) + self.N * i:comb(self.nAtoms + 1, 2) + self.N * (i + 1)]) ** 2,axis=1)
                    + np.sum(np.abs(finalStates[:, indexes]) ** 2, axis=1))
        EE = np.abs(finalStates[:, 2]) ** 2

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        self.dynamicsEE = EE
        self.finalStates = finalStates

        return self.dynamicsl2, self.dynamicsl1, EE

    def computeDynamics(self, ts, initialState):
        if self.es is None or self.vs is None:
            self.computeEigens()

        ns2 = np.zeros((self.nAtoms, len(ts)))
        ns1 = np.zeros((self.nAtoms, len(ts)))
        EE = np.zeros(len(ts))
        indexes = []
        iter = 0
        for t in ts:
            finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
            for j in range(len(self.Hamiltonian)):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]), initialState)
            for i in range(self.nAtoms):
                for j in range(self.nAtoms):
                    if i == j:
                        continue
                    elif i < j:
                        indexes.append(int(self.nAtoms + i * self.nAtoms - (i - 1) / 2 + j - 2 * i - 1))
                    else:
                        indexes.append(int(self.nAtoms + j * self.nAtoms - (j - 1) / 2 + i - 2 * j - 1))
                EE[iter] = np.abs(finalState[2]) ** 2
                ns2[i, iter] = np.abs(finalState[i]) ** 2
                ns1[i, iter] = (np.sum(np.abs(finalState[(comb(self.nAtoms + 1, 2) + self.N* i):(comb(self.nAtoms + 1, 2) + self.N * (i + 1))]) ** 2)
                              + np.sum(np.abs(finalState[indexes]) ** 2))

            iter += 1
            if iter % 100 == 0:
                print(iter)

        self.dynamicsl2 = ns2
        self.dynamicsl1 = ns1
        self.dynamicsEE = EE
        return self.dynamicsl2, self.dynamicsl1, EE

    def computeSiteDynamics(self, ts, initialState):
        if self.finalStates is None:
            self.computeDynamics2_0(ts, initialState)
        nt = len(ts)
        N = self.N

        resultE = np.zeros((nt, N))
        resultG = np.zeros((nt, N))
        resultGG = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            for k in range(self.N):
                indeces = []
                for l in range(self.nAtoms):
                    indeces.append(int(comb(self.nAtoms + 1, 2) + l * self.N + k))
                resultE[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)
                indeces = []
                for l in range(self.N):
                    if l >= k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))
                    else:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + l * self.N - l * (l - 1) / 2 + k - l))
                resultG[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)
                indeces = []
                for l in range(self.N):
                    if l == k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))

                    resultGG[i, k] = np.sum(np.abs(self.finalStates[i, indeces]) ** 2)
        self.resultE = resultE
        self.resultG = resultG
        self.resultGG = resultGG
        return resultE, resultG, resultGG

    def computeWavepacketDynamics(self, ts, initialState):  # how the wavepackt evolves in the waveguide
        if self.es is None or self.vs is None:
            self.computeEigens()

        nt = len(ts)
        N = self.N

        resultE = np.zeros((nt, N))
        resultG = np.zeros((nt, N))

        for i in range(nt):
            t = ts[i]
            finalState = np.zeros(len(self.Hamiltonian), dtype=complex)
            for j in range(len(self.Hamiltonian)):
                finalState += np.exp(-1j * self.es[j] * t) * self.vs[:, j] * np.dot(np.conjugate(self.vs[:, j]),
                                                                                    initialState)

            for k in range(self.N):
                indeces = []
                for l in range(self.nAtoms):
                    indeces.append(int(comb(self.nAtoms + 1, 2) + l * self.N + k))
                resultE[i, k] = np.sum(np.abs(finalState[indeces]) ** 2)
                indeces = []
                for l in range(self.N):
                    if l >= k:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + k * self.N - k * (k - 1) / 2 + l - k))
                    else:
                        indeces.append(int(comb(self.nAtoms + 1, 2) + self.nAtoms * self.N + l * self.N - l * (l - 1) / 2 + k - l))
                resultG[i, k] = np.sum(np.abs(finalState[indeces]) ** 2)

        return resultE, resultG

    def computeIPR(self):
        if self.vs is None:
            self.computeEigens()

        IPR = np.zeros(len(self.Hamiltonian))
        for i in range(len(self.Hamiltonian)):
            v = self.vs[:, i]
            for j in range(len(self.Hamiltonian)):
                IPR[i] += np.abs(v[j]) ** 4

        self.IPR = IPR
        return IPR

    def lookUpState(self, index):   # Index ---> atom and cavity numbers, index starts with 0, atoms/cavities with 1
        atoms = []
        cavities = []
        tempIndex = 0
        if index < self.nAtoms:
            atoms.append(index + 1)
            return atoms, cavities
        elif index < self.nAtoms + comb(self.nAtoms, 2):
            tempIndex = index - self.nAtoms
            for i in range(self.nAtoms):
                for j in range(i + 1, self.nAtoms):
                    atomIndex = i * self.nAtoms - i * (i - 1) / 2 + j - 2 * i - 1
                    if tempIndex == atomIndex:
                        atoms.append(i + 1)
                        atoms.append(j + 1)
                        return atoms, cavities
        elif index < self.nAtoms + comb(self.nAtoms + 1, 2) + self.N * self.nAtoms:
            tempIndex = index - self.nAtoms - comb(self.nAtoms, 2)
            q, mod = divmod(tempIndex, self.N)
            atoms.append(q + 1)
            cavities.append(mod + 1)
            return atoms, cavities
        else:
            tempIndex = index - self.nAtoms - comb(self.nAtoms + 1, 2) - self.N * self.nAtoms
            for i in range(self.N):
                for j in range(i, self.N):
                    cavityIndex = i * self.N - i * (i - 1) / 2 + j - i
                    if tempIndex == cavityIndex:
                        cavities.append(i + 1)
                        cavities.append(j + 1)
                        return atoms, cavities

    def saveGA(self, filename):
        dump(self, filename, compress=3)

    @classmethod
    def loadGA(cls, filename):
        return load(filename)










