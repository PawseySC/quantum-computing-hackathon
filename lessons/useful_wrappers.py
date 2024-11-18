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
# import plotly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.linalg import expm

# penny lane quantum computing library
import pennylane as qml
from pennylane import numpy as np
qml.drawer.use_style('sketch')

# quatnum music visualisation 
import qmuvi
from qiskit import QuantumCircuit

# set some paramters for plotting 
hfont = {'fontname':'Helvetica'}
plt.ioff() # let the interactive plot take over matplotlib interaction
plt.tight_layout()

def PlotPennyLaneHisto(results, plottitle : str = ''):
    with plt.xkcd():
    # Based on "Stove Ownership" from XKCD by Randall Munroe
    # https://xkcd.com/418/

        # fig, ax = plt.subplots()
        fig = plt.figure(figsize=(7.5,7.5))
        ax = fig.add_axes((0.2, 0.2, 0.5, 0.5))
        ax.spines[['top', 'right']].set_visible(False)
        labels = [rf'$|{k}>$' for k in results.keys()]
        xvals = np.arange(len(results.keys()))
        yvals = [results[k] for k in results.keys()]
        ax.bar(labels, yvals, facecolor='DarkOrange', edgecolor='Gold', linewidth=4)
        ax.tick_params(axis='x', labelrotation=80)
        # ax.set_xlabel('Qubit States', **hfont)
        # ax.set_ylabel('Count', **hfont)
        # ax.set_title(plottitle, **hfont)
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
        display(Markdown('## Simulating Circuit'))
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

        display(Markdown('## Simulating Circuit'))
        display(Markdown(f'You have asked to simulate a circuit with {num_qubits} and {num_gates} running {num_measurements} measurements'))
        display(Markdown(f'* Memory: This would require {2**(mem-30):.4f} GB of memory, or roughly {2**(mem-30)/8:.4f} laptops'))
        display(Markdown(f'* Operations: This would require ${np.exp(flops):.4f}$ Floating Point Operations, taking roughly {time:.4f} seconds on a laptop'))
        maxlognqubit=9
        maxgates = 10
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
    fig, ax = qml.draw_mpl(circuit, show_all_wires=True)(qubits, hqubits, cnotqubits)
    fig.show()
    if (num_measurements > max_nshots) or (num_qubits > max_nqubits):
        display(Markdown('## WARNING'))
        display(Markdown('You have asked to simulate either too many measurements (shots) or too many qubits. ***NOT SIMULATING***'))
        display(Markdown(f'* *Number of measurements should be $<{max_nshots}$ and requested:* {num_measurements}'))
        display(Markdown(f'* *Number of qubits should be $<{max_nqubits}$ and requested:* {num_qubits}. This would require {2**(num_qubits-30)} GB of memory and take a long time to simulate'))
    else:
        # now you try running more shots by running the circuit again. 
        results = circuit(qubits, hqubits, cnotqubits, shots=num_measurements)
        # here plot the results
        PlotPennyLaneHisto(results, f'Measurment results from {num_measurements} measurements')

def _maketarget(rows, cols, irand):
    if (rows < 2): rows = 2
    if (cols < 2): cols = 2
    x_target, y_target = rows/2, cols/2
    if (irand):
        x_target, y_target = np.random.randint(low = 1, high = 5, size = 2)

    x = [j + 1 for i in range(rows) for j in range(cols)]
    y = [i + 1 for i in range(rows) for j in range(cols)]
    return rows, cols, x_target, y_target, x, y

def GroversGrid(rows: int = 4, cols : int = 4, irand : bool = True, 
               msize = 50 ):
    '''
    @brief generate a grid and have people try to find the target
    '''
    display(Markdown('## Find the target!'))
    display(Markdown('Click on the circles and see if you can find the hidden target. If the circle turns Green, you have found it!'))
    display(Markdown('To restart, simply `shift+enter` on the cell.'))
    colorset = {'Hit': '#bae2be', 'Miss': 'White', 'Off': '#a3a7e4'}
    bbox = dict(boxstyle="round", fc="white", ec='black')
    def update_point(trace, points, selector):
        c = list(scatter.marker.color)
        s = list(scatter.marker.size)
        for i in points.point_inds:
            if x[i] == x_target and y[i] == y_target:  
                c[i] = colorset['Hit'] 
            else:
                c[i] = colorset['Miss']
            s[i] = 50
        with figwidget.batch_update():
            scatter.marker.color = c
            scatter.marker.size = s    
            ntries = list(scatter.marker.color).count(colorset['Miss'])
            if colorset['Hit'] in c:
                figwidget.update_layout(title = f"\nSuccess! In {ntries+1} tries.")
            else:
                figwidget.update_layout(title = f"\nNumber of misses = {ntries+1} of {len(c)}.")

    # initialize the target state
    rows, cols, x_target, y_target, x, y = _maketarget(rows, cols, irand)

    # create interative plotly figure
    figwidget = go.FigureWidget([go.Scatter(x=x, y=y, mode='markers')])
    scatter = figwidget.data[0]
    colors = [colorset['Off']] * (rows * cols)
    scatter.marker.color = colors
    scatter.marker.size = [msize] * (rows * cols)
    figwidget.layout.hovermode = 'closest'
    # update the layout 
    figwidget.update_layout(
        width=(2.05*msize)*(rows+1),  
        height=(2.05*msize)*(cols+1),
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1,
            range=[0, cols + 1],
            showticklabels=False,  
            tickvals=[],          
        ),
        yaxis=dict(
            range=[0, rows + 1],
            showticklabels=False,  
            tickvals=[],           
        ),
        plot_bgcolor='white',  
        paper_bgcolor='white',
        title=f"Where's the hidden target?",
    )
    figwidget.update_traces(marker=dict(
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                            )
    scatter.on_click(update_point)
    return figwidget

def GroverSearch(rows: int = 4, cols : int = 4, irand : bool = True, 
                 figwidth : int = 500, figheight : int = 750, 
                 jitter_strength : float = 0.0015):
    '''
    @brief show a grover's search where data is displayed in a grid and try to show how size of target increased by grovers search

    '''
    display(Markdown('## Make the target bigger!'))
    display(Markdown('By adjusting the slider you can see how the phase difference between the desired target and all the rest can change the system'))
    display(Markdown('You should be able to find the point at which the maximum probability (and area covered by the target) is much larger than the rest of the data points. Try seeing where you get the biggest target circle.'))
    display(Markdown('You should see in the total probability of finding the target that the probability depends on time when you measure the quantum circuit as well. So there is both a ideal phase and time to maximise the desired outcome.'))
    display(Markdown('To restart, simply `shift+enter` on the cell.'))

    colorset = {'Hit': '#bae2be', 'Miss': 'DarkOrange', 'Off': '#a3a7e4'}
    rows, cols, x_target, y_target, x, y = _maketarget(rows, cols, irand)
    n = rows * cols     
    target = n/2+1
    if irand: target = np.random.randint(low = 0, high = n)
    
    def prob(phase, ts, target, n):
        G = np.ones((n, n)) - np.eye(n)
        psi = (1/np.sqrt(n)) * np.ones(n, dtype=np.complex128)
        psi[0] *= np.exp(1j * phase)
        return [np.abs((expm(-1j * t * G) @ psi)[target])**2 for t in ts]
    
    def target_term(phase, t, n):
        return np.exp(1j * (n-1) * t)*[(1/np.sqrt(n))*np.exp(-1j * phase)]
    
    def non_target_term(phase, t, n):
        return (1/np.sqrt(n)) * (n-1) * np.exp(1j * (n-1) * t)*(np.exp(-1j * n * t) - 1)*(1/(n))*np.exp(-1j * phase)
    
    ts = np.arange(0, 1, 0.01)
    phis = np.linspace(0, 2 * np.pi, 100)  
    
    fig = make_subplots(
        rows=2, cols=2, shared_xaxes='columns',
        subplot_titles=("Probability of Measuring the Target", "Relative Probability", "Amplitude Contributions", None),
        specs=[
            [{"type": "xy"}, {"type": "xy", "rowspan": 2}], 
            [{"type": "xy"}, None]                           
        ],
    )
    
    
    phi = 0
    
    colors = [colorset['Miss']] * n
    colors[target] = colorset['Hit']  
    
    marker_sizes = n * [int(500 / n)]
        
    y_prob = prob(0, ts, 0, n)
    y_prob_jittered = y_prob + np.random.normal(0, jitter_strength, size=len(y_prob))
    
    y_target_term = np.abs(target_term(0, ts, n)) * np.real(target_term(0, ts, n))
    y_target_term_jittered = y_target_term + np.random.normal(0, jitter_strength, size=len(y_target_term))
    
    y_non_target_term = np.abs(non_target_term(0, ts, n)) * np.imag(non_target_term(0, ts, n))
    y_non_target_term_jittered = y_non_target_term + np.random.normal(0, jitter_strength, size=len(y_non_target_term))
    
    trace1 = go.Scatter(
        x=ts,
        y=y_prob_jittered,
        mode='lines',
        line=dict(color=colorset['Hit'], width=4, dash='solid'),
        name="Probability to Measure the Target",
        line_shape='spline'
    )
    
    trace2 = go.Scatter(
        x=ts,
        y=y_target_term_jittered,
        mode='lines',
        line=dict(color=colorset['Hit'], width=4, dash='solid'),
        name="Amplitude from the Target",
            line_shape='spline'
    )
    
    trace3 = go.Scatter(
        x=ts,
        y=y_non_target_term_jittered,
        mode='lines',
        line=dict(color=colorset['Miss'], width=4, dash='dash'),
        name="Amplitude From Other States",
        line_shape='spline'
    )
    
    trace4 = go.Scatter(
        x=x, y=y, mode='markers',
        name = "Relative Probability", 
        marker=dict(size=marker_sizes, color=colors)
    )
    
    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=2, col=1)
    fig.add_trace(trace3, row=2, col=1)
    fig.add_trace(trace4, row=1, col=2)
    
    steps = []
    for phi in phis:
        y_prob = prob(phi, ts, 0, n)
        y_prob_jittered = y_prob + np.random.normal(0, jitter_strength, size=len(y_prob))
    
        y_target_term = np.abs(target_term(phi, ts, n)) * np.real(target_term(phi, ts, n))
        y_target_term_jittered = y_target_term + np.random.normal(0, jitter_strength, size=len(y_target_term))
    
        y_non_target_term = np.abs(non_target_term(phi, ts, n)) * np.imag(non_target_term(phi, ts, n))
        y_non_target_term_jittered = y_non_target_term + np.random.normal(0, jitter_strength, size=len(y_non_target_term))
    
        marker_sizes = np.empty(n, dtype=int)
        marker_sizes[:] = int(500 * prob(phi, [ts[np.argmax(prob(phi, ts, 1, n))]], 1, n)[0])
        marker_sizes[target] = int(500 * prob(phi, [ts[np.argmax(prob(phi, ts, 0, n))]], 0, n)[0])
        step = dict(
            method="update",
            args=[
                {
                    "y": [
                        y_prob_jittered,
                        y_target_term_jittered,
                        y_non_target_term_jittered,
                        y
                    ],
                    "marker": [
                        None,
                        None,
                        None,
                        {"size": marker_sizes, "color": colors, "linecolor": 'DarkSlateGrey', "linewidth": 2}
                    ]
                }
            ],
            label=f"{np.degrees(phi):.0f}°"  
        )
        steps.append(step)
    
    sliders = [dict(
        active=0,  
        pad={"t": 50},
        steps=steps,
        currentvalue={"prefix": "Phase angle between target and non-target states: "}
    )]
    
    fig.update_yaxes(range=[0, 2.0/np.sqrt(n)], row=1, col=1)
    fig.update_yaxes(range=[-1.2/np.sqrt(n), 1.2/np.sqrt(n)], row=2, col=1)
    
    
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    
    #fig.update_xaxes(title_text='time', row=2, col=1)
    
    
    fig.update_layout(
        sliders=sliders,
        width=2 * figwidth, height=figheight,
        showlegend=False,   
        margin=dict(t=50),   
        font=dict(family="Comic Sans MS, sans-serif", size=16, color="black"),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
        yaxis=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
        xaxis3=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
        yaxis3=dict(showgrid=True, zeroline=True, zerolinewidth=2, zerolinecolor='Black'),
    )
    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')),)
    
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, row=1, col=1)
    
    xaxis_domain = fig.layout.xaxis2.domain
    yaxis_domain = fig.layout.yaxis2.domain
    
    fig.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=xaxis_domain[0],
        y0=yaxis_domain[0],
        x1=xaxis_domain[1],
        y1=yaxis_domain[1],
        fillcolor="white",
        #line=dict(width=0),
        layer="below"
    )
    return fig

