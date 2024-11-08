# some helper function definitions
# again we won't go into the details here but you will need to run the cell
# generic plotting library 
import matplotlib
import matplotlib.pyplot as plt
import mpl_interactions.ipyplot as iplt
matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
# import interactive widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display, Markdown, Latex

import plotly.graph_objects as go

# penny lane quantum computing library
import pennylane as qml
from pennylane import numpy as np
qml.drawer.use_style('sketch')

# quatnum music visualisation 
import qmuvi
from qiskit import QuantumCircuit

hfont = {'fontname':'Helvetica'}

def PlotPennyLaneHisto(results, plottitle : str = ''):
    with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

        # fig, ax = plt.subplots()
        # plt.tight_layout()
        plt.ioff()
        fig = plt.figure(figsize=(7.5,7.5))
        ax = fig.add_axes((0.2, 0.2, 0.5, 0.5))
        ax.spines[['top', 'right']].set_visible(False)
        labels = [rf'$|{k}>$' for k in results.keys()]
        xvals = np.arange(len(results.keys()))
        yvals = [results[k] for k in results.keys()]
        ax.bar(labels, yvals, facecolor='DarkOrange', edgecolor='Gold', linewidth=4)
        ax.tick_params(axis='x', labelrotation=80)
        ax.set_xlabel('Qubit States', **hfont)
        ax.set_ylabel('Count', **hfont)
        ax.set_title(plottitle, **hfont)
        fig.show()

def _setupqubits(num_qubits, add_H, hqubits, add_CNOT, cnotqubits):
    qubits = [f'|{i}>' for i in range(1,num_qubits+1)]
    if not add_H:
        hqubits = []
    else:
        if hqubits == 'all':
            hqubits = qubits
        else:
            # parse the hqubit list to ensure that it works
            newhqubits = []
            for q in hqubits:
                if q >=0 and q<num_qubits:
                    newhqubits.append(qubits[q])
            if len(newhqubits)>0:
                hqubits = newhqubits 
            else:
                hqubits = qubits
    if not add_CNOT or len(qubits) == 1:
        cnotqubits = []
    else:
        if cnotqubits == 'default':
            cnotqubits = []
            for q in qubits[1:]:
                cnotqubits.append([qubits[0],q])

    return qubits, hqubits, cnotqubits 

def _reportsim(num_qubits, num_gates, ireport : bool):

    if ireport:
        mem = 8*2**(2*num_qubits-30)*(num_gates+1)
        flops = num_gates * (4*num_qubits-1)*(2*num_qubits)**2.0+(4*num_qubits-1)*(2*num_qubits)
        display(Markdown('# Simulating Circuit'))
        display(Markdown(f'You have asked to simulate a circuit with {num_qubits} and {num_gates}'))
        display(Markdown(f'* Memory: This would require {mem:.4f} GB of memory, or roughly {mem/8:.4f} laptops'))
        display(Markdown(f'* Operations: This would require {flops:.4e} Floating point operations, or roughly {flops/5e12/128*0.5:.4e} seconds on a laptop'))

def PlotSystemRequirements(num_qubits : int = 2, num_gates : int = 1, num_measurements : int = 1):
    with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/
        
        def memcalc(x,y):
            return np.log(8.0)+2*x*np.log(2.0)+np.log(y)
        def flopcalc(x,y,z):
            return np.log(y * (4*x-1)*(2*x)**2.0+(4*x-1)*(2*x))+np.log(z)
        mem = memcalc(num_qubits, num_gates)
        flops = flopcalc(num_qubits, num_gates, num_measurements)
        memlist = {
            'very small': np.log(0.01)+30.0*np.log(2),
            'Laptop': np.log(8)+30.0*np.log(2), 
            '100 laptops':np.log(800)+30.0*np.log(2),
            'A Supercomputer':np.log(1e6)+30.0*np.log(2),
            'All the storage on Earth':np.log(200e12)+30.0*np.log(2),
            'A few hundred Earths': np.log(1e16)+30.0*np.log(2),
        }
        flopslist = {
            '1 second on a laptop': 5e12/128*0.5,
            #'1 year on a laptop': 5e12/128*0.5*3.15e7,
            '1000 years on a laptop': 5e12/128*0.5*3.15e10,
             '1 second on a Supercomputer node':5e12,
             '1 second on Setonix \nsupercomputer':42e15,
             '1 second on Frontier supercomputer or \n 2 years on a laptop':1e18,
             '1 second on all Google \ncloud computing': 3.98 * 1e21,
#             '1 second on Earth\'s computers':1e23,
#             '1 year using Earth\'s computers':3e30,
        }
        time = np.exp(flops)/flopslist['1 second on a laptop']

        display(Markdown('# Simulating Circuit'))
        display(Markdown(f'You have asked to simulate a circuit with {num_qubits} and {num_gates} running {num_measurements} measurements'))
        display(Markdown(f'* Memory: This would require {2**(mem-30):.4f} GB of memory, or roughly {2**(mem-30)/8:.4f} laptops'))
        display(Markdown(f'* Operations: This would require ${np.exp(flops):.4f}$ Floating Point Operations, taking roughly {time:.4f} seconds on a laptop'))
        maxlognqubit=9
        maxgates = 10
        plt.ioff()
        plt.tight_layout()
        x = 2**np.arange(maxlognqubit)
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,12))
        plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1, hspace=0.4)
        # ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        ax1.plot(x, memcalc(x, num_gates), linewidth = 4, zorder = 2, color='cornflowerblue', marker='None')
        ax1.scatter([num_qubits], [memcalc(num_qubits, num_gates)], 
                   zorder = 3, facecolor='Navy', edgecolor='LightBlue', marker='o', s=100)
        ax1.plot([0,num_qubits], [mem, mem], 
                linewidth = 2, zorder = 1, color='lightblue', marker='None', linestyle='dashed')
        ax1.plot([num_qubits,num_qubits], [-2, mem], 
                linewidth = 2, zorder = 1, color='lightblue', marker='None', linestyle='dashed')
        ax1.set_xscale('log')
        ax1.set_xlim([0,2**(maxlognqubit+0.1)])
        ax1.set_ylim([-2, np.max([mem*1.1, (memlist['All the storage on Earth'])])])

        ax1.set_yticks([memlist[k] for k in memlist.keys()], 
                      labels=memlist.keys()) 
        ax1.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256],
                      labels=['1', '2', '4', '8', '16', '32', '64', '128', '256']
                     ) 
        ax1.set_xlabel('Number of Qubits')
        #ax1.set_ylabel('Amount of Memory')
        ax1.set_title('How much does memory does it take to \nsimulate on a classical computer?')
        # fig.show()


        # fig2 = plt.figure()
        # ax2 = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        ax2.plot(x, flopcalc(x, num_gates, num_measurements), linewidth = 4, zorder = 2, color='darkorange', marker='None')
        ax2.scatter([num_qubits], [flopcalc(num_qubits, num_gates, num_measurements)], 
                   zorder = 3, facecolor='darkgoldenrod', edgecolor='gold', marker='o', s=100)
        
        ax2.plot([0,num_qubits], [flops, flops], 
                linewidth = 2, zorder = 1, color='gold', marker='None', linestyle='dashed')
        ax2.plot([num_qubits, num_qubits], [-10, flops], 
                linewidth = 2, zorder = 1, color='gold', marker='None', linestyle='dashed')
        ax2.set_xscale('log')
        ax2.set_yticks([np.log(flopslist[k]) for k in flopslist.keys()], 
                      labels=flopslist.keys())
        ax2.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256],
                      labels=['1', '2', '4', '8', '16', '32', '64', '128', '256']
                     ) 
        ax2.set_xlim([0,2**(maxlognqubit+0.1)])
        ax2.set_ylim([-10, np.max([flops*1.1, np.log(flopslist[list(flopslist.keys())[-1]])])])

        ax2.set_xlabel('Number of Qubits')
        #ax2.set_ylabel('Amount of Time')
        ax2.set_title('How long does it take to simulate \non a classical computer?')
        ax1.tick_params(axis='y', labelrotation=-5)
        ax2.tick_params(axis='y', labelrotation=-5)

        fig.show()


def MakeCircuit(num_qubits : int = 1, 
                add_H : bool = True, 
                add_CNOT : bool = True,
                hqubits : str = 'all',
                cnotqubits : str = 'default', 
                report_system_requirements : bool = False,
               ):
    """
    Construct a circuit with a certain number of qubits
    """
    qubits, hqubits, cnotqubits = _setupqubits(num_qubits, add_H, hqubits, add_CNOT, cnotqubits)
    num_gates = len(hqubits) + len(cnotqubits)
    # circuit     
    dev = qml.device("default.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit(qubits, hqubits, cnotqubits):
        for q in hqubits:
            qml.Hadamard(wires=q)
        if add_CNOT and len(qubits)>1: 
            for q in cnotqubits:
                qml.CNOT(q)
        return qml.counts(all_outcomes=True)

    _reportsim(num_qubits, num_gates, report_system_requirements)
    # plotting circuit
    plt.ioff()
    fig, ax = qml.draw_mpl(circuit, show_all_wires=True)(qubits, hqubits, cnotqubits)
    fig.show()
    

def MakeAndRunCircuit(num_measurements :int = 100, 
                      num_qubits : int = 1, 
                      add_H : bool = True, 
                      add_CNOT : bool = True,
                      hqubits : str = 'all',
                      cnotqubits : str = 'default', 
                      report_system_requirements : bool = False,
                     ):
    """
    Construct a circuit with a certain number of qubits and run a certain number of shots
    """
    max_nshots = 10000
    max_nqubits = 30
    qubits, hqubits, cnotqubits = _setupqubits(num_qubits, add_H, hqubits, add_CNOT, cnotqubits)
    num_gates = len(hqubits) + len(cnotqubits)
    # circuit     
    dev = qml.device("default.qubit", wires=qubits)
    @qml.qnode(dev)
    def circuit(qubits, hqubits, cnotqubits):
        for q in hqubits:
            qml.Hadamard(wires=q)
        if add_CNOT and len(qubits)>1: 
            for q in cnotqubits:
                qml.CNOT(q)
        return qml.counts(all_outcomes=True)

    _reportsim(num_qubits, num_gates, report_system_requirements)
    # plotting circuit
    plt.ioff()
    fig, ax = qml.draw_mpl(circuit, show_all_wires=True)(qubits, hqubits, cnotqubits)
    fig.show()
    if (num_measurements > max_nshots) or (num_qubits > max_nqubits):
        display(Markdown('# WARNING'))
        display(Markdown('You have asked to simulate either too many measurements (shots) or too many qubits. ***NOT SIMULATING***'))
        display(Markdown(f'* *Number of measurements should be $<{max_nshots}$ and requested:* {num_measurements}'))
        display(Markdown(f'* *Number of qubits should be $<{max_nqubits}$ and requested:* {num_qubits}. This would require {2**(num_qubits-30)} GB of memory and take a long time to simulate'))
    else:
        # now you try running more shots by running the circuit again. 
        results = circuit(qubits, hqubits, cnotqubits, shots=num_measurements)
        # here plot the results
        PlotPennyLaneHisto(results, f'Measurment results from {num_measurements} measurements')
