"""
Automatización de sistemas de distribución
Proyecto 2
Autores: Juan Sebastián Caicedo - Daniel García
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
    print("PROYECTO 3 - AUTOMATIZACIÓN DE SISTEMAS DE DISTRIBUCIÓN\n")
    print("Authors: Juan Sebastián Caicedo & Daniel García")
    kwargs = get_initial_data()
    S = [0.3]*8+[0.6]*8+[1]*8
    while True:
        try:
            reactive_energy = []
            active_energy = []
            reactive_energy_excess = []
            LossP_total = []
            PG = [0]*6+[1]*12+[0]*6
            option = display_main_menu()
            match option:
                case "1" | "1.":
                    PG = np.zeros(24)
                    for i, s in enumerate(S):
                        system = ElectricalSystem(PG[i], s, x0 = np.array([1, 0, 0]), **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system.energy_results()
                        loss_p = system.LossP()
                        LossP_total.append(loss_p)
                        reactive_energy.append(Reactive_Energy)
                        active_energy.append(Active_Energy)
                        reactive_energy_excess.append(Reactive_Energy_Excess)
                    print(f"\n{display_values(active_energy, reactive_energy, reactive_energy_excess, LossP_total)}\n")
                    financial_data = Financial_Analysis(active_energy, reactive_energy, reactive_energy_excess, LossP_total)
                    print(financial_data.display_prices())

                case "2" | "2.":
                    PG = np.zeros(24)
                    for i, s in enumerate(S):
                        system = ElectricalSystem(PG[i], s, optimized = False, **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system.energy_results()
                        optimized_results = system.optimize_Qc()
                        V2, theta2, Qc = optimized_results
                        system2 = ElectricalSystem(PG[i], s, np.array([V2, theta2, Qc]), optimized = True, **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system2.energy_results()
                        loss_p = system2.LossP()
                        LossP_total.append(loss_p)
                        reactive_energy.append(Reactive_Energy)
                        active_energy.append(Active_Energy)
                        reactive_energy_excess.append(Reactive_Energy_Excess)
                    print(f"\n{display_values(active_energy, reactive_energy, reactive_energy_excess, LossP_total)}\n")
                    financial_data = Financial_Analysis(active_energy, reactive_energy, reactive_energy_excess, LossP_total)
                    print(financial_data.display_prices())

                case "3" | "3.":
                    PG = np.array(PG)*0.5
                    for i, s in enumerate(S):
                        system = ElectricalSystem(PG[i], s, x0 = np.array([1, 0, 0]), **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system.energy_results()
                        loss_p = system.LossP()
                        LossP_total.append(loss_p)
                        reactive_energy.append(Reactive_Energy)
                        active_energy.append(Active_Energy)
                        reactive_energy_excess.append(Reactive_Energy_Excess)
                    print(f"\n{display_values(active_energy, reactive_energy, reactive_energy_excess, LossP_total)}\n")
                    financial_data = Financial_Analysis(active_energy, reactive_energy, reactive_energy_excess, LossP_total)
                    print(financial_data.display_prices())

                case "4" | "4.":
                    PG = np.array(PG)*0.5
                    for i, s in enumerate(S):
                        system = ElectricalSystem(PG[i], s, optimized = False, **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system.energy_results()
                        optimized_results = system.optimize_Qc()
                        V2, theta2, Qc = optimized_results
                        system2 = ElectricalSystem(PG[i], s, np.array([V2, theta2, Qc]), optimized = True, **kwargs)
                        Active_Energy, Reactive_Energy, Reactive_Energy_Excess = system2.energy_results()
                        loss_p = system2.LossP()
                        LossP_total.append(loss_p)
                        reactive_energy.append(Reactive_Energy)
                        active_energy.append(Active_Energy)
                        reactive_energy_excess.append(Reactive_Energy_Excess)
                    print(f"\n{display_values(active_energy, reactive_energy, reactive_energy_excess, LossP_total)}\n")
                    financial_data = Financial_Analysis(active_energy, reactive_energy, reactive_energy_excess, LossP_total)
                    print(financial_data.display_prices())

                case "5" | "5.":
                    sys.exit()
               
                case _:
                    print("Invalid option, try again...")
                    raise ValueError
        except ValueError:
            continue

        

def display_values(active_energy, reactive_energy, reactive_energy_excess, LossP_total):
    """
    Tabla de valores de energía
    """  
    active_energy = sum(active_energy)
    reactive_energy = sum(reactive_energy)
    reactive_energy_excess = sum(reactive_energy_excess)
    LossP_total = sum(LossP_total)
    table = np.array([
        ["Tipo de energía", "Valor"],
        ["Energía activa (kWh/día)", "{:,.2f}".format(active_energy)],
        ["Energía reactiva (kWAr/día)", "{:,.2f}".format(reactive_energy)],
        ["Exceso de energía reactiva\n(kWAr/día)", "{:,.2f}".format(reactive_energy_excess)],
        ["Pérdidas de potencia\nactiva (kWh/día)", "{:,.2f}".format(LossP_total*100*1e3)]
        ])
    return tabulate(table[1:], headers=table[0], tablefmt="grid")

def get_initial_data():       
    while True:
        try: 
            data = input("Ingrese los valores de RL, XL y FP separados por coma (,)\nSi desea los valores por defecto (9.3197, 3.72, 0.8) oprima ENTER: ")
            data = data.strip()
            matches = re.search(r"^(\d+\.\d+), *(\d+\.\d+), *([0-1]+\.\d+)$", data)
            if re.search(r"^$", data):
                fields = {}
                break
            elif matches:
                RL, XL, PF = float(matches.group(1)), float(matches.group(2)), float(matches.group(3))
                if (RL > 0) and (XL > 0) and (PF > 0 and PF <= 1.0):
                    fields = {"RL": RL, "XL": XL, "PF": PF}
                    break
                else:
                    raise ValueError
            else:
                raise ValueError
        except ValueError:
            print("Los datos ingresados son inválidos, inténtelo nuevamente")
            continue
        except AttributeError:
            continue
    return fields

def display_main_menu():
    print('MAIN MENU\n')
    print("1. Pago mensual por transporte de la energía reactiva en exceso")
    print("2. Optimización de inyección de potencia reactiva en el nodo 2")
    print("3. Caso de carga industrial con páneles solares")
    print("4. Optimización para el caso de carga industrial con páneles solares")
    print("5. Exit Program\n")
    option = input("Select one of the options presented above: ")
    return option


class ElectricalSystem:
    def __init__(self, PG, S, x0 = np.array([1, 0, 0]), optimized = False, RL=9.3197, XL=3.72, PF=0.8):
        self.Zbase = 132.25  # ohms
        self.V1 = 1  # PU
        self.theta_1 = 0  # rad
        self.RL = RL / self.Zbase  # PU
        self.XL = XL / self.Zbase  # PU
        self.PF = PF
        self.Sbase = 100  # MVA
        self.S = S * 100 / self.Sbase  # PU
        self.PD = self.S * self.PF  # PU
        self.QD = np.sqrt(self.S**2 - self.PD**2)  # PU
        self.PG = PG
        self.x0 = x0  # Initial guess
        self.optimized = optimized
    
    def line_current(self, x0, optimized = False):
        V2, theta_2, Qc = x0
        if not optimized:
            for _ in range(10):
                J = np.conj((self.PD - self.PG + 1j * (self.QD)) / (V2 * np.exp(1j * theta_2)))
                V2 = np.abs(self.V1 * np.exp(1j * self.theta_1) - (self.RL + 1j * self.XL) * J)
                theta_2 = np.angle(self.V1 * np.exp(1j * self.theta_1) + (self.RL + 1j * self.XL) * J)
            return J, V2, theta_2
        else:
            J = np.conj((self.PD - self.PG + 1j * (self.QD - Qc)) / (V2 * np.exp(1j * theta_2)))
            return J, V2, theta_2
    
    def LossP(self):
        J, _, _ = self.line_current(self.x0, optimized = self.optimized)
        return np.abs(J)**2 * self.RL
    
    def LossQ(self):
        J, _, _ = self.line_current(self.x0, optimized = self.optimized)
        return np.abs(J)**2 * self.XL
    
    def PG1(self):
        P_loss = self.LossP()
        return P_loss + self.PD - self.PG
    
    def QG1(self):
        V2, theta2, Qc = self.x0
        Q_loss = self.LossQ()
        return Q_loss + self.QD - Qc
    
    def energy_results(self):
        active_energy_Load = self.PD * self.Sbase * 1e3  # kWh/h
        reactive_energy_Load = self.QD * self.Sbase * 1e3  # kVArh/h
        active_energy = self.PG1() * self.Sbase * 1e3
        reactive_energy = self.QG1() * self.Sbase * 1e3
        fp_bus1 = active_energy / np.sqrt(active_energy**2 + reactive_energy**2)
        reactive_energy_excess = reactive_energy - 0.5 * abs(active_energy) if reactive_energy > 0.5 * abs(active_energy) else 0
        return active_energy, reactive_energy, reactive_energy_excess


    """
    Optimización de potencia reactiva
    """

    def objective(self, x0):
        V2, theta2, Qc = x0
        J = np.conj((self.PD - self.PG + 1j * (self.QD - Qc)) / (V2 * np.exp(1j * theta2)))
        return self.RL * np.abs(J)**2

    def constraints_eq(self, x):
        V2, theta2, Qc = x
        J = np.conj((self.PD - self.PG + 1j * (self.QD - Qc)) / (V2 * np.exp(1j * theta2)))
        kvl = V2 * np.exp(1j * theta2) - self.V1 * np.exp(1j * self.theta_1) + (self.RL + 1j * self.XL) * J
        # Condición de factor de potencia
        # Las restricciones deben retornar 0 para ser consideradas satisfechas en 'eq' y >0 para 'ineq'
        Ceq = np.array([np.real(kvl), np.imag(kvl)])  
        return Ceq
    
    def constraints_ineq(self, x):
        V2, theta2, Qc = x
        J = np.conj((self.PD - self.PG + 1j * (self.QD - Qc)) / (V2 * np.exp(1j * theta2)))
        fp = ((self.PD+self.RL*abs(J)**2)/(((self.PD+self.RL*abs(J)**2)**2+(self.QD+self.XL*abs(J)**2-Qc)**2)**(1/2)))-0.9
        return fp

    def optimize_Qc(self):
        x0 = self.x0
        restrictions = [{'type': 'ineq', 'fun': self.constraints_ineq},
                        {'type': 'eq', 'fun': self.constraints_eq}]
        
        results = minimize(self.objective, x0, constraints=restrictions)
        return results.x
    
    
    
class Financial_Analysis:

    def __init__(self, active_energy, reactive_energy, reactive_energy_excess, LossP_total):
        self.M = 12
        self.Td = 200 #COP/kWh
        self.energy_cost = 800 #COP/kWh
        self.active_energy = active_energy
        self.reactive_energy = reactive_energy
        self.reactive_energy_excess = reactive_energy_excess
        self.LossP_total = LossP_total

    def REE_payment(self):
        return 365 * np.sum(self.reactive_energy_excess) * self.M * self.Td / 1e9 # Milles de millones de COP por año
    
    def AE_payment(self):
        return 365 * np.sum(self.active_energy) * self.energy_cost / 1e9# Milles de millones de COP por año
    
    def ratio(self):
        return self.REE_payment() / self.AE_payment()
    
    def total_payment(self):
        return self.AE_payment() + self.REE_payment() # Milles de millones de COP por año
    
    def Loss_payment(self):
        return 365 * np.sum(self.LossP_total)*100*1e3 * self.energy_cost / 1e9 # Milles de millones de COP por año

    def display_prices(self):
        REEP = self.REE_payment()
        AEP = self.AE_payment()
        ratio = self.ratio()
        total_payment = self.total_payment()
        loss_payment = self.Loss_payment()
        table = np.array([
            ["Tipos de pagos", "Precio"],
            ["Pago por exceso de\nenergía reactiva\nfacturada (G COP/año)", "${:,.2f}".format(REEP)],
            ["Pago por energía\nactiva (G COP/año)", "${:,.2f}".format(AEP)],
            ["Tasa de pago de energía\nreactiva sobre pago de\nenergía activa", ratio],
            ["Pago total\n(G COP/año)", "${:,.2f}".format(total_payment)],
            ["Pagos por pérdidas\n(G COP/año)", "${:,.2f}".format(loss_payment)]
            ])
        return tabulate(table[1:], headers=table[0], tablefmt="grid")

if __name__ == '__main__':
    main()