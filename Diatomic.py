# Imports and Global Settings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import shutil
import time

debugMode = True
thisdir = pathlib.Path.cwd()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "monospace",
    "savefig.format": "pdf"
})

class Atom:
    """Atom Class"""

    def __init__(self, mass, position, velocity, acceleration):
        if (debugMode):
            if (not isinstance(mass, float)):
                raise RuntimeError("Invalid mass. Must be float type.")
            if (mass <= 0):
                raise RuntimeError("Invalid mass. Must be greater than 0.")
            if (not isinstance(position, np.ndarray) or not isinstance(velocity, np.ndarray) or not isinstance(acceleration, np.ndarray)):
                raise RuntimeError("Invalid position, velocity or acceleration. Must be NumPy arrays.")
            if (len(position) != 3 or len(velocity) != 3 or len(acceleration) != 3):
                raise RuntimeError("Invalid position, velocity or acceleration. Must have 3 elements.")
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

    def getMass(self):
        return self.mass

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return self.velocity

    def getAcceleration(self):
        return self.acceleration

    def setMass(self, mass):
        if (debugMode):
            if (not isinstance(mass, float)):
                raise RuntimeError("Invalid mass. Must be float type.")
            if (mass <= 0):
                raise RuntimeError("Invalid mass. Must be greater than 0.")
        self.mass = mass

    def setPosition(self, position):
        if (debugMode):
            if (not isinstance(position, np.ndarray)):
                raise RuntimeError("Invalid position. Must be a NumPy array.")
            if (len(position) != 3):
                raise RuntimeError("Invalid position. Must have 3 elements.")
        self.position = position

    def setVelocity(self, velocity):
        if (debugMode):
            if (not isinstance(velocity, np.ndarray)):
                raise RuntimeError("Invalid velocity. Must be a NumPy array.")
            if (len(velocity) != 3):
                raise RuntimeError("Invalid velocity. Must have 3 elements.")
        self.velocity = velocity

    def setAcceleration(self, acceleration):
        if (debugMode):
            if (not isinstance(acceleration, np.ndarray)):
                raise RuntimeError("Invalid acceleration. Must be a NumPy array.")
            if (len(acceleration) != 3):
                raise RuntimeError("Invalid acceleration. Must have 3 elements.")
        self.acceleration = acceleration

class Potential:
    """Potential Super Class"""

    def __init__(self):
        pass

    def getPotentialEnergy(self):
        pass

    def getForce(self):
        pass

class Harmonic(Potential):
    """Harmonic Potential Sub Class"""

    def __init__(self, re, k):
        if (debugMode):
            if (not isinstance(re, float) or not isinstance(k, float)):
                raise RuntimeError("Invalid params. Must be float type.")
            if (re <= 0 or k <= 0):
                raise RuntimeError("Invalid params. Must be greater than 0.")
        self.re = re
        self.k = k

    def getPotentialEnergy(self, r):
        if (debugMode):
            if (not isinstance(r, np.ndarray)):
                raise RuntimeError("Invalid separation distance. Must be a Numpy array.")
        lengthOfR = np.linalg.norm(r)
        return 0.5 * self.k * (lengthOfR - self.re)**2

    def getForce(self, r):
        if (debugMode):
            if (not isinstance(r, np.ndarray)):
                raise RuntimeError("Invalid separation distance. Must be a Numpy array.")
        lengthOfR = np.linalg.norm(r)
        f_x = -self.k * (lengthOfR - self.re) * r[0] / lengthOfR
        f_y = -self.k * (lengthOfR - self.re) * r[1] / lengthOfR
        f_z = -self.k * (lengthOfR - self.re) * r[2] / lengthOfR
        return np.array([f_x, f_y, f_z])

class Morse(Potential):
    """Morse Potential Class"""

    def __init__(self, re, a, De):
        if (debugMode):
            if (not isinstance(re, float) or not isinstance(a, float) or not isinstance(De, float)):
                raise RuntimeError("Invalid params. Must float type.")
            if (re <= 0 or a <= 0 or De <= 0):
                raise RuntimeError("Invalid params. Must be greater than 0.")
        self.re = re
        self.a = a
        self.De = De

    def getPotentialEnergy(self, r):
        if (debugMode):
            if (not isinstance(r, np.ndarray)):
                raise RuntimeError("Invalid separation distance. Must be a Numpy array.")
        lengthOfR = np.linalg.norm(r)
        return self.De * (1 - np.exp(-self.a * (lengthOfR - self.re)))**2

    def getForce(self, r):
        if (debugMode):
            if (not isinstance(r, np.ndarray)):
                raise RuntimeError("Invalid separation distance. Must be a Numpy array.")
        lengthOfR = np.linalg.norm(r)
        prefactor = -2 * self.De * self.a
        exponential = np.exp(-self.a * (lengthOfR - self.re))
        f_x = prefactor * (exponential - exponential**2) * r[0] / lengthOfR
        f_y = prefactor * (exponential - exponential**2) * r[1] / lengthOfR
        f_z = prefactor * (exponential - exponential**2) * r[2] / lengthOfR
        return np.array([f_x, f_y, f_z])

class Integrator:
    """Integrator Class"""

    def __init__(self):
        pass

    def validateInput(self, atoms, potential, deltaT):
        if (debugMode):
            if (not isinstance(atoms, np.ndarray)):
                raise RuntimeError("Invalid atoms. Must be a NumPy array.")
            if (len(atoms) != 2):
                raise RuntimeError("Invalid no. atoms. Must be 2.")
            if (not isinstance(atoms[0], Atom)):
                raise RuntimeError("Invalid atom. Must be Atom type.")
            if (not issubclass(type(potential), Potential)):
                raise RuntimeError("Invalid potential. Must be Potential type.")
            if (not isinstance(deltaT, float)):
                raise RuntimeError("Invalid deltaT. Must be float type.")
            if (deltaT <= 0):
                raise RuntimeError("Invalid deltaT. Must be greater than 0.")

    def rungeKutta4(self, atoms, potential, deltaT):
        """Fourth-Order Runge-Kutta Integrator"""

        self.validateInput(atoms, potential, deltaT)

        # k_1 and l_1
        currentVelocities = np.array([atoms[0].getVelocity(), atoms[1].getVelocity()])
        nextVelocities = np.copy(currentVelocities)
        currentPositions = np.array([atoms[0].getPosition(), atoms[1].getPosition()])
        nextPositions = np.copy(currentPositions)
        f_0 = potential.getForce(currentPositions[0] - currentPositions[1])
        f_1 = -f_0
        k_10 = deltaT * currentVelocities[0]
        k_11 = deltaT * currentVelocities[1]
        l_10 = deltaT * f_0
        l_11 = deltaT * f_1

        # k_2 and l_2
        nextVelocities[0] = currentVelocities[0] + 0.5 * l_10
        nextVelocities[1] = currentVelocities[1] + 0.5 * l_11
        k_20 = deltaT * nextVelocities[0]
        k_21 = deltaT * nextVelocities[1]
        nextPositions[0] = currentPositions[0] + 0.5 * k_10
        nextPositions[1] = currentPositions[1] + 0.5 * k_11
        f_0 = potential.getForce(nextPositions[0] - nextPositions[1])
        f_1 = -f_0
        l_20 = deltaT * f_0
        l_21 = deltaT * f_1

        # k_3 and l_3
        nextVelocities[0] = currentVelocities[0] + 0.5 * l_20
        nextVelocities[1] = currentVelocities[1] + 0.5 * l_21
        k_30 = deltaT * nextVelocities[0]
        k_31 = deltaT * nextVelocities[1]
        nextPositions[0] = currentPositions[0] + 0.5 * k_20
        nextPositions[1] = currentPositions[1] + 0.5 * k_21
        f_0 = potential.getForce(nextPositions[0] - nextPositions[1])
        f_1 = -f_0
        l_30 = deltaT * f_0
        l_31 = deltaT * f_1

        # k_4 and l_4
        nextVelocities[0] = currentVelocities[0] + l_30
        nextVelocities[1] = currentVelocities[1] + l_31
        k_40 = deltaT * nextVelocities[0]
        k_41 = deltaT * nextVelocities[1]
        nextPositions[0] = currentPositions[0] + k_30
        nextPositions[1] = currentPositions[1] + k_31
        f_0 = potential.getForce(nextPositions[0] - nextPositions[1])
        f_1 = -f_0
        l_40 = deltaT * f_0
        l_41 = deltaT * f_1

        # Consolidation
        nextVelocities[0] = currentVelocities[0] + (l_10 + 2 * l_20 + 2 * l_30 + l_40) / 6
        nextVelocities[1] = currentVelocities[1] + (l_11 + 2 * l_21 + 2 * l_31 + l_41) / 6
        nextPositions[0] = currentPositions[0] + (k_10 + 2 * k_20 + 2 * k_30 + k_40) / 6
        nextPositions[1] = currentPositions[1] + (k_11 + 2 * k_21 + 2 * k_31 + k_41) / 6
        atoms[0].setVelocity(nextVelocities[0])
        atoms[1].setVelocity(nextVelocities[1])
        atoms[0].setPosition(nextPositions[0])
        atoms[1].setPosition(nextPositions[1])
        return atoms

    def velocityVerlet(self, atoms, potential, deltaT):
        """Velocity-Verlet Integrator"""

        self.validateInput(atoms, potential, deltaT)

        # Consolidation
        f_0 = potential.getForce(atoms[0].getPosition() - atoms[1].getPosition())
        f_1 = -f_0
        atoms[0].setAcceleration(f_0 / atoms[0].getMass())
        atoms[1].setAcceleration(f_1 / atoms[1].getMass())
        r_0 = atoms[0].getPosition() + atoms[0].getVelocity() * deltaT + 0.5 * atoms[0].getAcceleration() * deltaT**2
        r_1 = atoms[1].getPosition() + atoms[1].getVelocity() * deltaT + 0.5 * atoms[1].getAcceleration() * deltaT**2
        atoms[0].setPosition(r_0)
        atoms[1].setPosition(r_1)
        f_0 = potential.getForce(atoms[0].getPosition() - atoms[1].getPosition())
        f_1 = -f_0
        v_0 = atoms[0].getVelocity() + 0.5 * (atoms[0].getAcceleration() + f_0 / atoms[0].getMass()) * deltaT
        v_1 = atoms[1].getVelocity() + 0.5 * (atoms[1].getAcceleration() + f_1 / atoms[1].getMass()) * deltaT
        atoms[0].setVelocity(v_0)
        atoms[1].setVelocity(v_1)
        return atoms

class MolecularSystem:
    """Molecular System Class"""

    def __init__(self, atoms, potential):
        if (debugMode):
            if (not isinstance(atoms, np.ndarray)):
                raise RuntimeError("Invalid atoms. Must be a NumPy array.")
            if (len(atoms) != 2):
                raise RuntimeError("Invalid no. atoms. Must be 2.")
            if (not isinstance(atoms[0], Atom)):
                raise RuntimeError("Invalid atom. Must be Atom type.")
            if (not issubclass(type(potential), Potential)):
                raise RuntimeError("Invalid potential. Must be Potential type.")
        self.atoms = atoms
        self.potential = potential

    def evolve(self, integratorName, deltaT):
        if (debugMode):
            if (not hasattr(Integrator, integratorName) or not callable(getattr(Integrator, integratorName))):
                raise RuntimeError("Invalid integrator name. Must be a callable Integrator method.")
            if (not isinstance(deltaT, float)):
                raise RuntimeError("Invalid deltaT. Must be float type.")
            if (deltaT <= 0):
                raise RuntimeError("Invalid deltaT. Must be greater than 0.")
        integrator = getattr(Integrator(), integratorName)
        self.atoms = integrator(self.atoms, self.potential, deltaT)

    def getAtomicSeparation(self):
        return {"Atom1-Atom2": np.linalg.norm(self.atoms[0].getPosition() - self.atoms[1].getPosition())}

    def getAtomicVelocity(self):
        return {"Atom1": self.atoms[0].getVelocity(), "Atom2": self.atoms[1].getVelocity()}

    def getEnergies(self):
        kin = 0.0
        pot = 0.0
        for atom in self.atoms:
            kin += 0.5 * atom.getMass() * np.linalg.norm(atom.getVelocity())**2
        r = self.atoms[0].getPosition() - self.atoms[1].getPosition()
        pot = self.potential.getPotentialEnergy(r)
        return {"KE": kin, "PE": pot, "TE": kin + pot}

def simulate(filepath, system, integratorName, N, deltaT):
    if (debugMode):
        if (not isinstance(filepath, pathlib.Path)):
            raise RuntimeError("Invalid filepath. Must be Path type.")
        if (not isinstance(system, MolecularSystem)):
            raise RuntimeError("Invalid system. Must be MolecularSystem type.")
        if (not hasattr(Integrator, integratorName) or not callable(getattr(Integrator, integratorName))):
            raise RuntimeError("Invalid integratorName. Must be a callable Integrator method.")
        if (not isinstance(N, int)):
            raise RuntimeError("Invalid N. Must be int type.")
        if (N <= 0):
            raise RuntimeError("Invalid N. Must be greater than 0.")
        if (not isinstance(deltaT, float)):
            raise RuntimeError("Invalid deltaT. Must be float type.")
        if (deltaT <= 0):
            raise RuntimeError("Invalid deltaT. Must be greater than 0.")
    if (filepath.exists()):
        shutil.rmtree(filepath)
    filepath.mkdir(exist_ok = True)
    aspath = filepath / 'AtomicSeparations.csv'
    avdir = filepath / 'AtomicVelocities'
    avdir.mkdir(exist_ok = True)
    enpath = filepath / 'Energies.csv'
    for i in range(0, N, 1):
        system.evolve(integratorName, deltaT)
        pd.DataFrame(system.getAtomicSeparation(), index = [0]).to_csv(aspath, mode = 'a', header = not aspath.exists(), index = False)
        avdict = system.getAtomicVelocity()
        v_x = 0.0
        v_y = 0.0
        v_z = 0.0
        for atomName in avdict.keys():
            velocity = avdict[atomName]
            temp = {"v_x": velocity[0], "v_y": velocity[1], "v_z": velocity[2]}
            avpath = avdir / f'{atomName}.csv'
            pd.DataFrame(temp, index = [0]).to_csv(avpath, mode = 'a', header = not avpath.exists(), index = False)
        pd.DataFrame(system.getEnergies(), index = [0]).to_csv(enpath, mode = 'a', header = not enpath.exists(), index = False)
    return system

def validateFilepath(filepath):
    if (not isinstance(filepath, pathlib.Path)):
        raise RuntimeError("Invalid filepath. Must be Path type.")
    if (not filepath.exists()):
        raise RuntimeError("Invalid filepath. Does not exist.")

def plotEnergies(filepath, name):
    validateFilepath(filepath)
    if (name not in ["kp", "total", "all"]):
            raise RuntimeError("Invalid name. Must be either kp, total or all.")
    energies = pd.read_csv(filepath)
    xs = np.arange(0, len(energies.index), 1)
    if (name == "kp" or name == "all"):
        plt.plot(xs, energies["KE"], "r--", label = "Kinetic Energy")
        plt.plot(xs, energies["PE"], "g--", label = "Potential Energy")
    if (name == "total" or name == "all"):
        plt.plot(xs, energies["TE"], "b--", label = "Total Energy")
    plt.xlabel("$N$")
    plt.ylabel("Energy")
    plt.legend()
    plt.title("Energy vs. Iteration")
    plt.savefig(filepath.parent / 'EnergyVsIteration')
    plt.show()

def plotRunningAverageEnergies(filepath, name):
    validateFilepath(filepath)
    if (name not in ["kp", "total", "all"]):
            raise RuntimeError("Invalid name. Must be either kp, total or all.")
    energies = pd.read_csv(filepath)
    kes = energies["KE"]
    pes = energies["PE"]
    tes = energies["TE"]
    N = len(energies.index)
    raKin = np.zeros(N)
    raPot = np.zeros(N)
    raTot = np.zeros(N)
    raKin[0] = kes[0]
    raPot[0] = pes[0]
    raTot[0] = tes[0]
    for i in range(1, N, 1):
        raKin[i] = (raKin[i - 1] * i + kes.iat[i]) / (i + 1)
        raPot[i] = (raPot[i - 1] * i + pes.iat[i]) / (i + 1)
        raTot[i] = (raTot[i - 1] * i + tes.iat[i]) / (i + 1)
    xs = np.arange(0, N, 1)
    if (name == "kp" or name == "all"):
        plt.plot(xs, raKin, "r--", label = "Kinetic Energy")
        plt.plot(xs, raPot, "g--", label = "Potential Energy")
    if (name == "total" or name == "all"):
        plt.plot(xs, raTot, "b--", label = "Total Energy")
    plt.xlabel("$N$")
    plt.ylabel("Avergage Energy")
    plt.legend()
    plt.title("Average Energy vs. Iteration")
    plt.savefig(filepath.parent / 'AverageEnergyVsIteration')
    plt.show()

def plotAtomicSeparations(filepath, pairName):
    validateFilepath(filepath)
    if (debugMode):
        if (not isinstance(pairName, str)):
            raise RuntimeError("Invalid pair name. Must be str type")
    atomicSeparations = pd.read_csv(filepath)
    xs = np.arange(0, len(atomicSeparations.index), 1)
    plt.plot(xs, atomicSeparations[pairName], "r--", label = "Atomic Separation")
    plt.xlabel("$N$")
    plt.ylabel("Length")
    plt.legend()
    plt.title("Atomic Separation vs. Iteration")
    plt.savefig(filepath.parent / 'AtomicSeparationVsIteration')
    plt.show()

def calculateAcf(data):
    if (debugMode):
        if (not isinstance(data, np.ndarray)):
            raise RuntimeError("Invalid data. Must be a NumPy array.")
        if (not isinstance(data[0], float)):
            raise RuntimeError("Invalid data. Must be float type.")
    data_ = data - np.mean(data)
    acf = np.correlate(data_, data_, mode = "full")
    acf_ = acf[acf.size // 2:]
    return acf_

def calculateVacf(filedir, fileNames):
    validateFilepath(filedir)
    if (debugMode):
        if (not isinstance(fileNames, list)):
            raise RuntimeError("Invalid file names. Must be list type.")
        for name in fileNames:
            if (not isinstance(name, str)):
                raise RuntimeError("Invalid file names. Must be str type.")
    try:
        N = len(pd.read_csv(filedir / f'{fileNames[0]}.csv').index)
        vacf = np.zeros(calculateAcf(np.random.rand(N)).shape)
        for atomName in fileNames:
            dataFrame = pd.read_csv(filedir / f'{atomName}.csv')
            vacf += calculateAcf(dataFrame["v_x"].to_numpy()) + calculateAcf(dataFrame["v_y"].to_numpy()) + calculateAcf(dataFrame["v_z"].to_numpy())
        return vacf
    except Exception as err:
        raise err

def calculatePowerSpectrum(vacf):
    if (debugMode):
        if (not isinstance(vacf, np.ndarray)):
            raise RuntimeError("Invalid vacf. Must be a NumPy array.")
        if (not isinstance(vacf[0], float)):
            raise RuntimeError("Invalid vacf. Must be float type.")
    powerSpectrum = np.abs(np.fft.fft(vacf))**2
    return powerSpectrum

def plotPowerSpectrum(filepath, powerSpectrum, xlim, yscalelog = True):
    validateFilepath(filepath)
    if (debugMode):
        if (not isinstance(powerSpectrum, np.ndarray)):
            raise RuntimeError("Invalid power spectrum. Must be a NumPy array.")
        if (not isinstance(powerSpectrum[0], float)):
            raise RuntimeError("Invalid power spectrum. Must be float type.")
        if (not isinstance(xlim, list)):
            raise RuntimeError("Invalid xlim. Must be list type.")
        if (len(xlim) != 2):
            raise RuntimeError("Invalid xlim. Must have 2 elements.")
        if (not isinstance(xlim[0], float) or not isinstance(xlim[1], float)):
            raise RuntimeError("Invalid xlim. Must be float type.")
        if (not isinstance(yscalelog, bool)):
            raise RuntimeError("Invalid yscalelog. Must be bool type.")
    plt.plot(powerSpectrum, label = "Power Spectrum")
    plt.xlim(xlim)
    if (yscalelog):
        plt.yscale("log")
    plt.xlabel("$\omega$")
    plt.ylabel("$\log(P(\omega))$")
    plt.legend()
    plt.title("Power Spectrum")
    plt.savefig(filepath / 'PowerSpectrum')
    plt.show()
