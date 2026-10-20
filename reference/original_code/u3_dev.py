import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pennylane as qml

# === Config ===
SAVE_DIR = "/scratch90/chris/cnn_vqc_expressivity/single_qubit"
os.makedirs(SAVE_DIR, exist_ok=True)

# === Bloch Vector Conversion ===
def bloch_coords(state):
    a, b = state[0], state[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return [x, y, z]

# === Plotting ===
def plot_bloch_sphere(bloch_vectors, title, save_path):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(30, 45)

    # Axes
    x, y, z = np.array([[-1.5,0,0], [0,-1.5,0], [0,0,-1.5]])
    u, v, w = np.array([[3,0,0], [0,3,0], [0,0,3]])
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.05, color="black", linewidth=0.5)

    # Labels
    ax.text(0, 0, 1.7, r"$|0\rangle$", color="black", fontsize=16)
    ax.text(0, 0, -1.9, r"$|1\rangle$", color="black", fontsize=16)
    ax.text(1.9, 0, 0, r"$|+\rangle$", color="black", fontsize=16)
    ax.text(-1.7, 0, 0, r"$|-\rangle$", color="black", fontsize=16)
    ax.text(0, 1.7, 0, r"$|i+\rangle$", color="black", fontsize=16)
    ax.text(0, -1.9, 0, r"$|i-\rangle$", color="black", fontsize=16)

    # Points
    ax.scatter(
        bloch_vectors[:, 0],
        bloch_vectors[:, 1],
        bloch_vectors[:, 2],
        c='#e29d9e',
        alpha=0.3
    )

    ax.set_title(title, pad=20)

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# === Devices ===
dev = qml.device("default.qubit", wires=1)

# === QNodes ===
@qml.qnode(dev)
def apply_u3(theta, phi, lam):
    qml.U3(theta, phi, lam, wires=0)
    return qml.state()

@qml.qnode(dev)
def apply_rx(theta):
    qml.RX(theta, wires=0)
    return qml.state()

@qml.qnode(dev)
def apply_ry(theta):
    qml.RY(theta, wires=0)
    return qml.state()

@qml.qnode(dev)
def apply_rz(theta):
    qml.RZ(theta, wires=0)
    return qml.state()

# === Samplers ===
def sample_haar_u3(n_samples=1000):
    coords = []
    for _ in range(n_samples):
        theta = np.arccos(1 - 2 * np.random.rand())
        phi = 2 * np.pi * np.random.rand()
        lam = 2 * np.pi * np.random.rand()
        state = apply_u3(theta, phi, lam)
        coords.append(bloch_coords(state))
    return np.array(coords)

def sample_rotation_gate(qnode_fn, n_samples=200):
    coords = []
    thetas = np.linspace(0, 2 * np.pi, n_samples)
    for theta in thetas:
        state = qnode_fn(theta)
        coords.append(bloch_coords(state))
    return np.array(coords)


def state_to_bloch_vector(state):
    """Convert a 1-qubit state vector to a Bloch vector (x, y, z)."""
    a, b = state[0], state[1]
    x = 2 * np.real(np.conj(a) * b)
    y = 2 * np.imag(np.conj(a) * b)
    z = np.abs(a)**2 - np.abs(b)**2
    return np.array([x, y, z])

# === Device ===
dev = qml.device("default.qubit", wires=1)

def single_gate_rotation_bloch(gate_type, theta):
    @qml.qnode(dev)
    def circuit():
        qml.BasisState(np.array([0]), wires=0)  # Start in |0⟩

        if gate_type.lower() == "rx":
            qml.RX(theta, wires=0)
        elif gate_type.lower() == "ry":
            qml.RY(theta, wires=0)
        elif gate_type.lower() == "rz":
            qml.RZ(theta, wires=0)
        elif gate_type.lower() == "u3":
            # Use theta as shared input; phi and lam can be fixed arbitrarily here
            phi = np.pi / 3
            lam = np.pi / 5
            qml.U3(theta, phi, lam, wires=0)
        return qml.state()

    state = circuit()
    return state_to_bloch_vector(state)




# === Main ===
if __name__ == "__main__":
    plot_bloch_sphere(
        sample_haar_u3(1000),
        title="U3 Haar Sampled",
        save_path=os.path.join(SAVE_DIR, "u3_bloch.png")
    )

    plot_bloch_sphere(
        sample_rotation_gate(apply_rx),
        title="RX Sweep",
        save_path=os.path.join(SAVE_DIR, "rx_bloch.png")
    )

    plot_bloch_sphere(
        sample_rotation_gate(apply_ry),
        title="RY Sweep",
        save_path=os.path.join(SAVE_DIR, "ry_bloch.png")
    )

    plot_bloch_sphere(
        sample_rotation_gate(apply_rz),
        title="RZ Sweep",
        save_path=os.path.join(SAVE_DIR, "rz_bloch.png")
    )

    theta = np.pi / 3  # 60 degrees

    for gate in ["rx", "ry", "rz", "u3"]:
        bloch = single_gate_rotation_bloch(gate, theta)
        print(f"Bloch vector after {gate.upper()}(π/3): x={bloch[0]:.4f}, y={bloch[1]:.4f}, z={bloch[2]:.4f}")