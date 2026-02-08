import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
import os

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 2):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, 
                             angle=angle, **kwargs))

np.random.seed(42)

# 1. Generate "Cline" data (Curved distribution)
# Create a stretched, slightly curved distribution to simulate a genetic gradient
n_samples = 500
t = np.random.uniform(0, 3, n_samples)
x = t + np.random.normal(0, 0.1, n_samples)
y = 0.5 * t + 0.2 * t**2 + np.random.normal(0, 0.1, n_samples)
X = np.column_stack([x, y])

# 2. Fit GMM (Over-segmentation scenario) - e.g., 3 components for a single cline
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))

# Plot data points
ax.scatter(X[:, 0], X[:, 1], s=10, c='gray', alpha=0.5, label='Samples (Cline)')

# Plot GMM components
colors = ['#FF5733', '#33FF57', '#3357FF']
for i, (mean, cov) in enumerate(zip(gmm.means_, gmm.covariances_)):
    draw_ellipse(mean, cov, ax=ax, alpha=0.3, color=colors[i], label=f'Component {i+1}')
    ax.scatter(mean[0], mean[1], marker='x', s=100, c='black', zorder=10)

ax.set_title("Schematic: Over-segmentation of a Continuous Gradient", fontsize=14)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.6)

# Add annotation arrows
# Text pointing to the "cuts"
ax.annotate('Cut 1', xy=(0.8, 0.5), xytext=(0.2, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.annotate('Cut 2', xy=(2.0, 1.6), xytext=(2.5, 0.5),
            arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()

# Determine output directory relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
fig_dir = os.path.join(project_dir, 'fig')

# Create figure directory if it doesn't exist
os.makedirs(fig_dir, exist_ok=True)

# Save figure
output_path = os.path.join(fig_dir, '02a.over_segmentation_demo.png')
plt.savefig(output_path, dpi=300)
print(f"Figure generated: {output_path}")
