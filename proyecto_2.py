"""
Automatización de sistemas de distribución
Proyecto 2
Autor: Juan Sebastián Caicedo
"""
from pyfiglet import Figlet
from tabulate import tabulate
import numpy as np
from scipy.optimize import minimize
import sys
import re
import matplotlib.pyplot as plt
import numpy_financial as npf

def main():
    figlet = Figlet()
    output_title = figlet.setFont(font = 'big')
    output_title = figlet.renderText('Optimal Power Flow in Distribution Systems')
    print(f'{output_title}\n')
    print("Author: Juan Sebastián Caicedo")
    kwargs = get_initial_data()
    S = [0.3, 0.6, 1]
    while True:
        try:
            power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values = [], [], [], [], [], []
            option = display_main_menu()
            match option:
                case "1" | "1.":
                    for s in S:
                        system = ElectricalSystem(100, 0, 0, s, *kwargs)
                        losses, V2, Qc, theta_2 = system.solve("power loss")
                        table = system.data("power loss")
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
                        system = ElectricalSystem(0, 100, 0, s, *kwargs)
                        losses, V2, Qc, theta_2 = system.solve("power loss")
                        table = system.data("power loss")
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
                        system = ElectricalSystem(100, 0, 0, s, *kwargs)
                        losses, V2, Qc, theta_2 = system.solve("minimum voltage")
                        table = system.data("minimum voltage")
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
                        system = ElectricalSystem(0, 100, 0, s, *kwargs)
                        losses, V2, Qc, theta_2 = system.solve("minimum voltage")
                        table = system.data("minimum voltage")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)

                case "5" | "5.":
                    S_percentage, Z_percentage, I_percentage = get_percentages()
                    for s in S:
                        system = ElectricalSystem(S_percentage, Z_percentage, I_percentage,  s, *kwargs)
                        losses, V2, Qc, theta_2 = system.solve("power loss")
                        table = system.data("power loss")
                        print(f'\n{table}\n')
                        power += [system.PD]*8
                        power_losses += [losses]*8
                        reactive_power_losses += [system.QD]*8
                        Qc_values += [Qc]*8
                        V2_values += [V2]*8
                        theta_2_values += [theta_2]*8
                    graphics(power, power_losses, reactive_power_losses, Qc_values, V2_values, theta_2_values)
                    
                case "6" | "6.":
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
            data = input("Ingrese los valores de RL, XL y FP separados por coma (,)\nSi desea los valores por defecto (9.3197, 3.72, 0.8) oprima ENTER: ")
            data = data.strip()
            matches = re.search(r"^(\d+\.\d+), *(\d+\.\d+), *([0-1]+\.\d+)$", data)
            if re.search(r"^$", data):
                return []
            elif matches:
                RL, XL, PF = float(matches.group(1)), float(matches.group(2)), float(matches.group(3))
                if (RL > 0) and (XL > 0) and (PF > 0 and PF <= 1.0):
                    return [RL, XL, PF]
                else:
                    raise ValueError
            else:
                raise ValueError
        except ValueError:
            print("Los datos ingresados son inválidos, inténtelo nuevamente")
            continue
        except AttributeError:
            continue

def fraction_to_float(fraction_str):
        if '/' in fraction_str:
            numerator, denominator = map(int, fraction_str.split('/'))
            return float(numerator/denominator)
        else:
            return float(fraction_str)
    
def get_percentages():
    while True:
        try: 
            data = input("Ingrese los porcentajes de las cargas de potencia, impedancia y corriente constante: ")
            data = data.strip()
            matches = re.search(r"^(\d+(?:\/\d+)*), *(\d+(?:\/\d+)*), *(\d+(?:\/\d+)*)$", data)
            if matches:
                S_percentage, Z_percentage, I_percentage = fraction_to_float(matches.group(1)), fraction_to_float(matches.group(2)), fraction_to_float(matches.group(3))
                if S_percentage + Z_percentage + I_percentage != 100:
                    raise AttributeError
                else:
                    return S_percentage, Z_percentage, I_percentage
        except ValueError:
            print("Los datos ingresados son inválidos, inténtelo nuevamente")
            continue
        except AttributeError:
            print("La suma de los porcentajes no es igual a 100, inténtelo de nuevo.")
            continue

def display_main_menu():
    print('MAIN MENU\n')
    print("1. Constant Power & Minimum Power loss")
    print("2. Constant Impedance & Minimum Power loss")
    print("3. Constant Power & Minimum ΔV")
    print("4. Constant Impedance & Minimum ΔV")
    print("5. Constant loads by percentages & Minimum Power loss")
    print("6. Exit Program\n")
    option = input("Select one of the options presented above: ")
    return option


class ElectricalSystem:
    def __init__(self, S_percentage, Z_percentage, I_percentage, S, RL = 9.3197, XL = 3.72, PF = 0.8):
        self.S_percentage = S_percentage
        self.Z_percentage = Z_percentage
        self.I_percentage = I_percentage
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

    @property
    def S_percentage(self):
        return self._S_percentage
    
    @S_percentage.setter
    def S_percentage(self, S_percentage):
        if S_percentage <= 100:
            self._S_percentage = S_percentage
        else:
            raise AttributeError

    @property
    def Z_percentage(self):
        return self._Z_percentage
    
    @Z_percentage.setter
    def Z_percentage(self, Z_percentage):
        if Z_percentage <= 100:
            self._Z_percentage = Z_percentage
        else:
            raise AttributeError

    @property
    def I_percentage(self):
        return self._I_percentage
    
    @I_percentage.setter
    def I_percentage(self, I_percentage):
        if I_percentage <= 100:
            self._I_percentage = I_percentage
        else:
            raise AttributeError
    
    def line_current(self, case, x0 = np.array([1, 0, 0])):
        V2 = x0[0] #PU
        theta_2 = x0[1] #rad
        Qc = 0 if len(x0) < 3 else x0[2]
        if case == 'power':
            return np.conj((complex(self.PD, self.QD - Qc))/(V2*complex(np.cos(theta_2), np.sin(theta_2)))) #PU
        elif case == 'impedance':
            Q = self.S*np.sqrt(1-self.PF**2)
            ZD2 = 1/np.conj(complex(self.S*self.PF, Q))
            return V2*complex(np.cos(theta_2), np.sin(theta_2))/ZD2 + complex(0, Qc)/np.conj(V2*complex(np.cos(theta_2), np.sin(theta_2))) #PU
        elif case == 'current':
            return np.conj(complex(self.PD, self.QD)) + np.conj(complex(0, -Qc)) / np.conj(V2 * complex(np.cos(theta_2), np.sin(theta_2))) #PU

    def constraints(self, x0 = np.array([1, 0, 0])):
        V2 = x0[0] #PU
        theta_2 = x0[1] #rad
        JS, JZ, JI = self.line_current("power", x0), self.line_current("impedance", x0), self.line_current("current", x0)
        JS *= self.S_percentage/100
        JZ *= self.Z_percentage/100
        JI *= self.I_percentage/100
        kV1 = V2*complex(np.cos(theta_2), np.sin(theta_2)) - self.V1*complex(np.cos(self.theta_1), np.sin(self.theta_1)) + complex(self.RL, self.XL)*(JS + JZ + JI) #PU
        return np.array([np.real(kV1), np.imag(kV1)])
    
    def power_losses(self, x0):
        JS, JZ, JI = self.line_current("power", x0), self.line_current("impedance", x0), self.line_current("current", x0)
        return self.RL*(self.S_percentage*abs(JS)**2 + self.Z_percentage*abs(JZ)**2 + self.I_percentage*abs(JI)**2)/100
    
    def minimum_delta(self, x0):
        V2 = x0[0] #PU
        return abs(self.V1 - V2)
        
    def solve(self, objective, include_Qc = True):
        x0 = self.x0 if include_Qc else self.x0[:2]
        restrictions = {'type': 'eq', 'fun': lambda x: self.constraints(x)}
        if objective == "power loss":
            solution = minimize(lambda x: self.power_losses(x), x0, constraints=restrictions)
        elif objective == "minimum voltage":
            solution = minimize(self.minimum_delta, x0, constraints=restrictions)
        if include_Qc:
            return solution.fun, solution.x[0], solution.x[2], np.rad2deg(solution.x[1])
        else:
            return solution.fun, solution.x[0], 0, np.rad2deg(solution.x[1])

    def data(self, objective):
        load_type = f"{round(self.S_percentage)}% const. power\n" + \
            f"{round(self.Z_percentage)}% const. impedance\n" + \
            f"{round(self.I_percentage)}% const. current"
        Ploss, V2, Qc, theta_2 = self.solve(objective)
        PlossnC, V2nC, QcnC, theta_2_nC = self.solve(objective, include_Qc = False)
        financial_data = Financial_Analysis(Qc, Ploss, PlossnC)
        VPN = financial_data.VPN()
        TIR = financial_data.TIR()
        payback_time = financial_data.Payback_Time()
        CB = financial_data.CB()
        data_table = np.array([
            ["Load Type", load_type], 
            ["V2 (P.U.)", V2], 
            ["Theta_2 (deg)", theta_2], 
            ["Qc (P.U.)", Qc], 
            ["Pérdidas (P.U.)", Ploss],
            ["Pérdidas sin\ncompensación", PlossnC],
            ["VPN (COP)", "${:,.2f}".format(VPN)],
            ["TIR (%)", TIR],
            ["Payback Time (years)", payback_time],
            ["Beneficio/Costo", CB],
            ["S (MVA)", self.S*self.Sbase]
        ])
        return tabulate(data_table[1:], headers=data_table[0], tablefmt="grid")

class Financial_Analysis:

    def __init__(self, Qc, Compensation, No_Compensation):
        """
        Constants regarding the profitability of the reactive power compensation installation project
        """
        self.cost_per_MVAr = 95000 #USD
        self.anual_discount_rate = 0.1 
        self.energy_cost = 800 #COP/KWh
        self.TRM = 4000
        self.lifespan = 20 #years
        self.loss_factor = 29/60
        self.hours_per_year = 8760 #h/year
        self.Sbase = 100
        self.Qc = Qc*self.Sbase
        self.Compensation = Compensation*self.Sbase
        self.No_Compensation = No_Compensation*self.Sbase
        self.Investment = self.Qc*self.cost_per_MVAr*self.TRM
        """Cálculos para los indicadores"""
        Energy_losses_no_compensation = self.No_Compensation*1000*self.loss_factor*self.hours_per_year
        Energy_losses_compensation = self.Compensation*1000*self.loss_factor*self.hours_per_year
        Energy_losses_payments_no_compensation = Energy_losses_no_compensation*self.energy_cost
        Energy_losses_payments_compensation = Energy_losses_compensation*self.energy_cost
        self.Savings = Energy_losses_payments_no_compensation - Energy_losses_payments_compensation
    
    def VPN(self):
        Valor_Presente_Neto = -self.Investment
        for i in range(1,self.lifespan+1):
            Valor_Presente_Neto += self.Savings/(1+self.anual_discount_rate)**i
        return Valor_Presente_Neto
    
    def TIR(self):
        cash_flows = [self.Savings]*self.lifespan
        cash_flows.insert(0, -self.Investment)
        tir = npf.irr(cash_flows)
        return tir*100 #%

    def Payback_Time(self):
        years = 0
        balance = self.Investment
        for i in range(1,self.lifespan+1):
            flow_at_t = self.Savings/(1+self.anual_discount_rate)**i
            if balance >= flow_at_t:
                balance -= flow_at_t
                years += 1
            else:
                years += balance/flow_at_t
                break
        return years

    def CB(self):
        benefits = sum([self.Savings/(1+self.anual_discount_rate)**t for t in range(1, self.lifespan + 1)])
        return benefits/self.Investment

if __name__ == '__main__':
    main()
