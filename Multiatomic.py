# Imports and Global Settings
import copy
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

    def __init__(self, name, mass, position, velocity, acceleration):
        if (debugMode):
            if (not isinstance(name, str)):
                raise RuntimeError("Invalid name. Must be str type.")
            if (not isinstance(mass, float)):
                raise RuntimeError("Invalid mass. Must be float type.")
            if (mass <= 0):
                raise RuntimeError("Invalid mass. Must be greater than 0.")
            if (not isinstance(position, np.ndarray) or not isinstance(velocity, np.ndarray) or not isinstance(acceleration, np.ndarray)):
                raise RuntimeError("Invalid position, velocity or acceleration. Must be NumPy arrays.")
            if (len(position) != 3 or len(velocity) != 3 or len(acceleration) != 3):
                raise RuntimeError("Invalid position, velocity or acceleration. Must have 3 elements.")
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

    def getName(self):
        return self.name

    def getMass(self):
        return self.mass

    def getPosition(self):
        return self.position

    def getVelocity(self):
        return self.velocity

    def getAcceleration(self):
        return self.acceleration

    def setName(self, name):
        if (debugMode):
            if (not isinstance(name, str)):
                raise RuntimeError("Invalid name. Must be str type.")
        self.name = name

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

    def __init__(self, potential, **params):
        if (debugMode):
            if (not callable(potential)):
                raise RuntimeError("Invalid potential. Must be callable.")
        self.potential = potential
        self.params = params

    def getPotentialEnergy(self, atoms):
        return self.potential(atoms, self.params)

class Force:
    """Force Class"""

    def __init__(self):
        pass

    def validateInput(self, atoms, potential, h):
        if (debugMode):
            if (not isinstance(atoms, np.ndarray)):
                raise RuntimeError("Invalid atoms. Must be a NumPy array.")
            for atom in atoms:
                if (not isinstance(atom, Atom)):
                    raise RuntimeError("Invalid atoms. Must be Atom type.")
            if (not isinstance(potential, Potential)):
                raise RuntimeError("Invalid potential. Must be Potential type.")
            if (not isinstance(h, float)):
                raise RuntimeError("Invalid h. Must be float type.")
            if (h <= 0):
                raise RuntimeError("Invalid h. Must be greater than 0.")

    def calculateForcesOrder1(self, atoms, potential, h = 1e-08):
        """2-point Stencil (First-Order Accurate) Numerical Approximation"""

        self.validateInput(atoms, potential, h)

        n = len(atoms)
        forces = np.zeros((n, 3))
        for atomIndex in range(0, n, 1):
            for positionIndex in [0, 1, 2]:

                # Current Point
                current = copy.deepcopy(atoms)

                # Right Displacement
                right = copy.deepcopy(atoms)
                atom = right[atomIndex]
                position = atom.getPosition()
                position[positionIndex] += h
                atom.setPosition(position)
                right[atomIndex] = atom

                # Force Component
                forceComponent = -(potential.getPotentialEnergy(right) - potential.getPotentialEnergy(current)) / h
                forces[atomIndex, positionIndex] = forceComponent
        return forces

    def calculateForcesOrder2(self, atoms, potential, h = 1e-08):
        """3-point Stencil (Second-Order Accurate) Numerical Approximation"""

        self.validateInput(atoms, potential, h)

        # # Test Mock
        # r = atoms[0].getPosition() - atoms[1].getPosition()
        # lengthOfR = np.linalg.norm(r)
        # f_x = -1 * (lengthOfR - 1) * r[0] / lengthOfR
        # f_y = -1 * (lengthOfR - 1) * r[1] / lengthOfR
        # f_z = -1 * (lengthOfR - 1) * r[2] / lengthOfR
        # f = np.array([f_x, f_y, f_z])

        n = len(atoms)
        forces = np.zeros((n, 3))
        for atomIndex in range(0, n, 1):
            for positionIndex in [0, 1, 2]:

                # Left Displacement
                left = copy.deepcopy(atoms)
                atom = left[atomIndex]
                position = atom.getPosition()
                position[positionIndex] -= h
                atom.setPosition(position)
                left[atomIndex] = atom

                # Right Displacement
                right = copy.deepcopy(atoms)
                atom = right[atomIndex]
                position = atom.getPosition()
                position[positionIndex] += h
                atom.setPosition(position)
                right[atomIndex] = atom

                # Force Component
                forceComponent = -(potential.getPotentialEnergy(right) - potential.getPotentialEnergy(left)) / (2 * h)
                forces[atomIndex, positionIndex] = forceComponent

        # assert np.allclose(np.array(forces), np.array([f, -f]))
        # return np.array([f, -f])

        return forces

class Integrator:
    """Integrator Class"""

    def __init__(self):
        pass

    def validateInput(self, atoms, potential, deltaT):
        if (debugMode):
            if (not isinstance(atoms, np.ndarray)):
                raise RuntimeError("Invalid atoms. Must be a NumPy array.")
            for atom in atoms:
                if (not isinstance(atom, Atom)):
                    raise RuntimeError("Invalid atoms. Must be Atom type.")
            if (not issubclass(type(potential), Potential)):
                raise RuntimeError("Invalid potential. Must be Potential type.")
            if (not isinstance(deltaT, float)):
                raise RuntimeError("Invalid deltaT. Must be float type.")
            if (deltaT <= 0):
                raise RuntimeError("Invalid deltaT. Must be greater than 0.")

    def rungeKutta4(self, atoms, potential, deltaT):
        """Fourth-Order Runge-Kutta Integrator"""

        self.validateInput(atoms, potential, deltaT)

        n = len(atoms)

        # k_1 and l_1
        currentVelocities = np.zeros((n, 3))
        currentPositions = np.zeros((n, 3))
        for i in range(0, n, 1):
            currentVelocities[i, :] = atoms[i].getVelocity()
            currentPositions[i, :] = atoms[i].getPosition()
        nextVelocities = np.copy(currentVelocities)
        nextPositions = np.copy(currentPositions)
        forces = Force().calculateForcesOrder1(atoms, potential, 1e-06)
        k_1s = np.zeros((n, 3))
        l_1s = np.zeros((n, 3))
        for i in range(0, n, 1):
            k_1s[i, :] = deltaT * currentVelocities[i]
            l_1s[i, :] = deltaT * forces[i]

        # k_2 and l_2
        nextAtoms = np.copy(atoms)
        k_2s = np.zeros((n, 3))
        l_2s = np.zeros((n, 3))
        for i in range(0, n, 1):
            nextVelocities[i] = currentVelocities[i] + 0.5 * l_1s[i]
            k_2s[i, :] = deltaT * nextVelocities[i]
            nextPositions[i] = currentPositions[i] + 0.5 * k_1s[i]
            nextAtoms[i].setPosition(nextPositions[i])
        forces = Force().calculateForcesOrder1(nextAtoms, potential, 1e-06)
        for i in range(0, n, 1):
            l_2s[i, :] = deltaT * forces[i]

        # k_3 and l_3
        nextAtoms = np.copy(atoms)
        k_3s = np.zeros((n, 3))
        l_3s = np.zeros((n, 3))
        for i in range(0, n, 1):
            nextVelocities[i] = currentVelocities[i] + 0.5 * l_2s[i]
            k_3s[i, :] = deltaT * nextVelocities[i]
            nextPositions[i] = currentPositions[i] + 0.5 * k_2s[i]
            nextAtoms[i].setPosition(nextPositions[i])
        forces = Force().calculateForcesOrder1(nextAtoms, potential, 1e-06)
        for i in range(0, n, 1):
            l_3s[i, :] = deltaT * forces[i]

        # k_4 and l_4
        nextAtoms = np.copy(atoms)
        k_4s = np.zeros((n, 3))
        l_4s = np.zeros((n, 3))
        for i in range(0, n, 1):
            nextVelocities[i] = currentVelocities[i] + l_3s[i]
            k_4s[i, :] = deltaT * nextVelocities[i]
            nextPositions[i] = currentPositions[i] + k_3s[i]
            nextAtoms[i].setPosition(nextPositions[i])
        forces = Force().calculateForcesOrder1(nextAtoms, potential, 1e-06)
        for i in range(0, n, 1):
            l_4s[i, :] = (deltaT * forces[i])

        # Consolidation
        for i in range(0, n, 1):
            nextVelocities[i] = currentVelocities[i] + (l_1s[i] + 2 * l_2s[i] + 2 * l_3s[i] + l_4s[i]) / 6
            nextPositions[i] = currentPositions[i] + (k_1s[i] + 2 * k_2s[i] + 2 * k_3s[i] + k_4s[i]) / 6
            atoms[i].setVelocity(nextVelocities[i])
            atoms[i].setPosition(nextPositions[i])
        return atoms

    def velocityVerlet(self, atoms, potential, deltaT):
        """Velocity-Verlet Integrator"""

        self.validateInput(atoms, potential, deltaT)

        n = len(atoms)

        # Consolidation
        forces = Force().calculateForcesOrder2(atoms, potential, 1e-06)
        for i in range(0, n, 1):
            atoms[i].setAcceleration(forces[i] / atoms[i].getMass())
            r_i = atoms[i].getPosition() + atoms[i].getVelocity() * deltaT + 0.5 * atoms[i].getAcceleration() * deltaT**2
            atoms[i].setPosition(r_i)
        forces = Force().calculateForcesOrder2(atoms, potential, 1e-06)
        for i in range(0, n, 1):
            v_i = atoms[i].getVelocity() + 0.5 * (atoms[i].getAcceleration() + forces[i] / atoms[i].getMass()) * deltaT
            atoms[i].setVelocity(v_i)
        return atoms

class MolecularSystem:
    """Molecular System Class"""

    def __init__(self, atoms, potential):
        if (debugMode):
            if (not isinstance(atoms, np.ndarray)):
                raise RuntimeError("Invalid atoms. Must be a NumPy array.")
            for atom in atoms:
                if (not isinstance(atom, Atom)):
                    raise RuntimeError("Invalid atoms. Must be Atom type.")
            if (not isinstance(potential, Potential)):
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

    def getAtomNames(self):
        atomNames = []
        for atom in self.atoms:
            atomNames.append(atom.getName())
        return atomNames

    def getAtomicPosition(self):
        atomicPosition = {}
        for atom in self.atoms:
            key = atom.getName()
            value = atom.getPosition()
            atomicPosition[key] = value
        return atomicPosition

    def getAtomicSeparation(self):
        noAtoms = len(self.atoms)
        atomicSeparation = {}
        for i in range(0, noAtoms, 1):
            atomi = self.atoms[i]
            for j in range(i + 1, noAtoms, 1):
                atomj = self.atoms[j]
                key = f'{atomi.getName()}-{atomj.getName()}'
                value = np.linalg.norm(atomi.getPosition() - atomj.getPosition())
                atomicSeparation[key] = value
        return atomicSeparation

    def getAtomicVelocity(self):
        atomicVelocity = {}
        for atom in self.atoms:
            key = atom.getName()
            value = atom.getVelocity()
            atomicVelocity[key] = value
        return atomicVelocity

    def getEnergies(self):
        kin = 0.0
        pot = 0.0
        for i in range(0, len(self.atoms), 1):
            kin += 0.5 * self.atoms[i].getMass() * np.linalg.norm(self.atoms[i].getVelocity())**2
        pot = self.potential.getPotentialEnergy(self.atoms)
        return {"KE": kin, "PE": pot, "TE": kin + pot}

def sampleVelocities(kB, mass, temperature, noPoints):
    """Sample Atomic Velocities From a Maxwell-Boltzmann Distribution at a Given Temperature"""

    if (debugMode):
        if (not isinstance(kB, float)):
            raise RuntimeError("Invalid kB. Must be float type.")
        if (kB <= 0):
            raise RuntimeError("Invalid kB. Must be greater than 0.")
        if (not isinstance(mass, float)):
            raise RuntimeError("Invalid mass. Must be float type.")
        if (mass <= 0):
            raise RuntimeError("Invalid mass. Must be greater than 0.")
        if (not isinstance(temperature, float)):
            raise RuntimeError("Invalid temperature. Must be float type.")
        if (temperature <= 0):
            raise RuntimeError("Invalid temperature. Must be greater than 0.")
        if (not isinstance(noPoints, int)):
            raise RuntimeError("Invalid number of points. Must be int type.")
        if (noPoints <= 0):
            raise RuntimeError("Invalid number of points. Must be greater than 0.")
    sigma2 = kB * temperature / mass
    sigma = np.sqrt(sigma2)
    vs = np.random.normal(0.0, sigma, (noPoints, 3))

    # # Distribution Plot
    # sampledVelocities = np.linalg.norm(vs, axis = 1)
    # count, bins, ignored = plt.hist(sampledVelocities, 30, density = True, label = "Histogram")
    # fv = (1 / ((2 * np.pi * sigma2)**1.5)) * 4 * np.pi * bins**2 * np.exp(-0.5 * bins**2 / sigma2)
    # plt.plot(bins, fv, linewidth = 2, color = "r", label = "MB Distribution")
    # plt.xlabel("$v$")
    # plt.ylabel("$f(v)$")
    # plt.legend()
    # plt.title("MB-Distributed Sampled Speeds")
    # plt.show()

    return vs[:, 0], vs[:, 1], vs[:, 2]

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
    apdir = filepath / 'AtomicPositions'
    apdir.mkdir(exist_ok = True)
    aspath = filepath / 'AtomicSeparations.csv'
    avdir = filepath / 'AtomicVelocities'
    avdir.mkdir(exist_ok = True)
    enpath = filepath / 'Energies.csv'
    for i in range(0, N, 1):
        system.evolve(integratorName, deltaT)
        pd.DataFrame(system.getAtomicSeparation(), index = [0]).to_csv(aspath, mode = 'a', header = not aspath.exists(), index = False)

        # Store Atomic Position Velocity
        apdict = system.getAtomicPosition()
        avdict = system.getAtomicVelocity()
        for atomName in system.getAtomNames():
            position = apdict[atomName]
            velocity = avdict[atomName]
            tempr = {"r_x": position[0], "r_y": position[1], "r_z": position[2]}
            tempv = {"v_x": velocity[0], "v_y": velocity[1], "v_z": velocity[2]}
            appath = apdir / f'{atomName}.csv'
            avpath = avdir / f'{atomName}.csv'
            pd.DataFrame(tempr, index = [0]).to_csv(appath, mode = 'a', header = not appath.exists(), index = False)
            pd.DataFrame(tempv, index = [0]).to_csv(avpath, mode = 'a', header = not avpath.exists(), index = False)
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
        plt.ylabel("$\log(P(\omega))$")
    else:
        plt.ylabel("$P(\omega)$")
    plt.xlabel("$\omega$")
    plt.legend()
    plt.title("Power Spectrum")
    plt.savefig(filepath / 'PowerSpectrum')
    plt.show()

# # MB-Distributed Sampled Speeds
# kB = 3.166811563 * 10e-6 # Hartree / Kelvin
# mass = 32.0 # Dalton
# temperature = 100.0
# noPoints = 100000
# v_x, v_y, v_z = sampleVelocities(kB, mass, temperature, noPoints)

# # Normally-Distributed Sampled Velocity Components
# sigma2 = kB * temperature / mass
# sigma = np.sqrt(sigma2)
# count, bins, ignored = plt.hist(v_x, 30, density=True, label = "Histogram")
# plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * bins**2 / sigma**2), linewidth=2, color='r', label = "Normal Distribution")
# plt.xlabel("$v_x$")
# plt.ylabel("$f(v_x)$")
# plt.legend()
# plt.title("Normally-Distributed Sampled Velocity Components")
# plt.show()
