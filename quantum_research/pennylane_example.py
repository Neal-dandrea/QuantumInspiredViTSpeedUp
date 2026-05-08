import pennylane as qml
from pennylane import numpy as np  # PennyLane's numpy (supports autodiff)

# 1. Create a device (simulator)
dev = qml.device('default.qubit', wires=2)

# 2. Define a quantum function
@qml.qnode(dev)
def my_circuit(params):
    qml.RX(params[0], wires=0)  # Rotate qubit 0 around X-axis
    qml.RY(params[1], wires=1)  # Rotate qubit 1 around Y-axis
    qml.CNOT(wires=[0, 1])      # Entangle
    return qml.expval(qml.PauliZ(0))  # Measure Z expectation on qubit 0

# 3. Run it!
params = np.array([0.5, 0.3])
result = my_circuit(params)
print(f"Result: {result}")