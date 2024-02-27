"""
Automatización de sistemas de distribución
Proyecto 1
Autores: Juan Sebastián Caicedo - Daniel García
"""
from pyfiglet import Figlet
from tabulate import tabulate
import numpy as np
from scipy.optimize import minimize
import sys
import re
import matplotlib.pyplot as plt

def main():
    figlet = Figlet()
    output_title = figlet.setFont(font = 'big')
    output_title = figlet.renderText('Optimal Power Flow in Distribution Systems')
    print(f'{output_title}\n')
    print("Authors: Juan Sebastián Caicedo & Daniel García")
    RL, XL, PF = get_initial_data()
    S = [0.3, 0.6, 1]
    while True:
        try:
            power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values = [], [], [], [], [], []
            option = display_main_menu()
            match option:
                case "1" | "1.":
                    for s in S:
                        system = ElectricalSystem(RL, XL, PF, s)
                        table, losses, Qc, V2, theta_2 = system.solve("power", "power loss")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)
                    
                case "2" | "2.":
                    for s in S:
                        system = ElectricalSystem(RL, XL, PF, s)
                        table, losses, Qc, V2, theta_2 = system.solve("impedance", "power loss")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)
                    
                case "3" | "3.":
                    for s in S:
                        system = ElectricalSystem(RL, XL, PF, s)
                        table, losses, Qc, V2, theta_2 = system.solve("power", "minimum voltage")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)
                    
                case "4" | "4.":
                    for s in S:
                        system = ElectricalSystem(RL, XL, PF, s)
                        table, losses, Qc, V2, theta_2 = system.solve("impedance", "minimum voltage")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)
                    
                case "5" | "5.":
                    sys.exit()
                case _:
                    print("Invalid option, try again...")
                    raise ValueError
        except ValueError:
            continue

def graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values):
    time = np.arange(0, 24)
    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (15,4))

    #P vs t
    axs[0, 0].plot(time, power, 'tab:blue')
    axs[0, 0].set_title('Potencia activa\nen 24 horas\n$S_{base}: 100MVA$')
    axs[0, 0].set_xlabel('Tiempo (h)')
    axs[0, 0].set_ylabel('Potencia (P.U.)')

    # Plosses vs t
    axs[0, 1].plot(time, power_losses, 'tab:orange')
    axs[0, 1].set_title('Pérdidas de potencia\nen 24 horas\n$S_{base}: 100MVA$')
    axs[0, 1].set_xlabel('Tiempo (h)')
    axs[0, 1].set_ylabel('Pérdidas de potencia (P.U.)')

    # Q vs t
    axs[0, 2].plot(time, reactive_power_losses, 'tab:green')
    axs[0, 2].set_title('Potencia reactiva\nen 24 horas\n$S_{base}: 100MVA$')
    axs[0, 2].set_xlabel('Tiempo (h)')
    axs[0, 2].set_ylabel('Potencia reactiva (P.U.)')

    # Qc vs t
    axs[1, 0].plot(time, Qc_values, 'tab:red')
    axs[1, 0].set_title('Qc en 24 horas\n$S_{base}: 100MVA$')
    axs[1, 0].set_xlabel('Tiempo (h)')
    axs[1, 0].set_ylabel('Potencia reactiva (P.U.)')

    #V2 vs t
    axs[1, 1].plot(time, V2_values, 'tab:blue')
    axs[1, 1].set_title('V2 en 24 horas\n$V_{base}: 115 kV$')
    axs[1, 1].set_xlabel('Tiempo (h)')
    axs[1, 1].set_ylabel('Voltaje (P.U.)')

    #theta_2 vs t
    axs[1, 2].plot(time, theta_2_values, 'tab:green')
    axs[1, 2].set_title('$\Theta_2$ en 24 horas')
    axs[1, 2].set_xlabel('Tiempo (h)')
    axs[1, 2].set_ylabel('$\Theta_2$ (deg)')


    plt.tight_layout()
    plt.show()

def get_initial_data():       
    while True:
        try: 
            data = input("Ingrese los valores de RL, XL y FP separados por coma (,): ")
            data = data.strip()
            matches = re.search(r"^([0-9]+\.[0-9]+), *([0-9]+\.[0-9]+), *([0-1]+\.[0-9]+)$", data)
            if matches:
                RL, XL, PF = float(matches.group(1)), float(matches.group(2)), float(matches.group(3))
                if (RL > 0) and (XL > 0) and (PF > 0 and PF <= 1.0):
                    break
                else:
                    raise ValueError
            else:
                raise ValueError
        except ValueError:
            print("Los datos ingresados son inválidos, inténtelo nuevamente")
            continue
    return RL, XL, PF

def display_main_menu():
    print('MAIN MENU\n')
    print("1. Constant Power & Minimum Power loss")
    print("2. Constant Impedance & Minimum Power loss")
    print("3. Constant Power & Minimum ΔV")
    print("4. Constant Impedance & Minimum ΔV")
    print("5. Exit Program\n")
    option = input("Select one of the options presented above: ")
    return option


class ElectricalSystem:
    def __init__(self, RL, XL, PF, S):
        self.Zbase = 132.25 #ohms
        self.V1 = 1 #PU
        self.theta_1 = 0 #rad
        self.RL = RL/self.Zbase #PU
        self.XL = XL/self.Zbase #PU
        self.PF = PF
        self.Sbase = 100 #MVA
        self.S = S*100/self.Sbase #PU
        self.PD = self.S*self.PF #PU
        self.QD = np.sqrt(self.S**2-self.PD**2) #PU
        self.x0 = np.array([1, 0, 0]) #Initial guess

    def line_current(self, case, x0 = np.array([1, 0, 0])):
        V2 = x0[0] #PU
        theta_2 = x0[1] #rad
        Qc = x0[2] #PU
        if case == 'power':
            return np.conj((complex(self.PD, self.QD - Qc))/(V2*complex(np.cos(theta_2), np.sin(theta_2)))) #PU
        elif case == 'impedance':
            Q = self.S*np.sqrt(1-self.PF**2)
            ZD2 = 1/np.conj(complex(self.S*self.PF, Q))
            return V2*complex(np.cos(theta_2), np.sin(theta_2))/ZD2 + complex(0, Qc)/np.conj(V2*complex(np.cos(theta_2), np.sin(theta_2)))

    def constraints(self, case, x0 = np.array([1, 0, 0])):
        V2 = x0[0] #PU
        theta_2 = x0[1] #rad
        J = self.line_current(case, x0) 
        kV1 = V2*complex(np.cos(theta_2), np.sin(theta_2)) - self.V1*complex(np.cos(self.theta_1), np.sin(self.theta_1)) + complex(self.RL, self.XL)*J #PU
        return np.array([np.real(kV1), np.imag(kV1)])
    
    def power_losses(self, case , x0):
        J = self.line_current(case, x0) 
        return self.RL*abs(J)**2
    
    def minimum_delta(self, x0):
        V2 = x0[0] #PU
        return abs(self.V1 - V2)
        
    def solve(self, case, objective):
        restrictions = {'type': 'eq', 'fun': lambda x: self.constraints(case, x)}
        if objective == "power loss":
            solution = minimize(lambda x: self.power_losses(case, x), self.x0, constraints=restrictions)
        elif objective == "minimum voltage":
            solution = minimize(self.minimum_delta, self.x0, constraints=restrictions)
        data_table = np.array([
            ["OPF Type", f"Constant {case.capitalize()}"], 
            ["V2 (P.U.)", solution.x[0]], 
            ["Theta_2 (deg)", np.rad2deg(solution.x[1])], 
            ["Qc (P.U.)", solution.x[2]], 
            ["Pérdidas (P.U.)", solution.fun],
            ["S (MVA)", self.S*self.Sbase]
        ])
        return tabulate(data_table[1:], headers=data_table[0], tablefmt="grid"), solution.fun, solution.x[2], solution.x[0], np.rad2deg(solution.x[1])

if __name__ == '__main__':
    main()
