import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyArrow
from matplotlib.animation import FuncAnimation

data = loadmat('data_files/Mouse12-120806_stuff_simple_awakedata.mat')

positions_head = data['position'][:, 2:4]
positions_rear = data['position'][:, :2]
trackingtimes = data['trackingtimes'][0]
cellspikes = data['cellspikes'][0]
headangle = data['headangle'][0]



spiketrain = cellspikes[25][0]



# Define invalid entries (positions or heading)
invalid_mask = (
    np.any(positions_head == -1, axis=1) |
    np.any(positions_rear == -1, axis=1) |
    np.any(np.isnan(positions_head), axis=1) |
    np.any(np.isnan(positions_rear), axis=1) |
    np.isnan(headangle)
)

# Apply the mask to keep only valid entries
positions_head = positions_head[~invalid_mask]
positions_rear = positions_rear[~invalid_mask]
headangle = headangle[~invalid_mask]
trackingtimes = trackingtimes[~invalid_mask]



print(f'Shape of position: {positions_head.shape}')
print(f'Shape of trackingtimes: {trackingtimes.shape}')
print(f'Shape of headangle: {headangle.shape}')
print(f'Shape of spiketrain: {spiketrain.shape}')


print(f'Sample of position: {positions_head}')
print(f'Sample of trackingtimes: {trackingtimes}')
print(f'Sample of headangle: {headangle}')
print(f'Sample of spiketrain: {spiketrain}')



# Load your data here
# Assume you've already loaded these arrays:
# position (85504, 2), trackingtimes (85504,), headangle (85504,), spiketrain (27176,)

# Interpolate to uniform time base (30 fps)
fps = 30
t_start = trackingtimes[0]
t_end = trackingtimes[-1]
uniform_time = np.arange(t_start, t_end, 1000 / fps)  # timestamps in same units

# Interpolate position and headangle
interp_x = np.interp(uniform_time, trackingtimes, positions_head[:, 0])
interp_y = np.interp(uniform_time, trackingtimes, positions_head[:, 1])
interp_rear_x = np.interp(uniform_time, trackingtimes, positions_rear[:, 0])
interp_rear_y = np.interp(uniform_time, trackingtimes, positions_rear[:, 1])

interp_angle = np.interp(uniform_time, trackingtimes, np.nan_to_num(headangle, nan=0.0))

# Interpolate heading at spike times
spike_headangles = np.interp(spiketrain, trackingtimes, np.nan_to_num(headangle, nan=0.0))

# Set up figure
fig, (ax_pos, ax_hd) = plt.subplots(1, 2, figsize=(10, 5))

# Ax for position
mouse_dot, = ax_pos.plot([], [], 'ro')
rear_dot, = ax_pos.plot([], [], 'bo')   # Blue dot for rear
body_line, = ax_pos.plot([], [], 'gray', lw=2)
arrow = FancyArrowPatch((0, 0), (0, 0), arrowstyle='->', mutation_scale=10, color='blue')
ax_pos.add_patch(arrow)
ax_pos.set_xlim(np.nanmin(interp_x) - 5, np.nanmax(interp_x) + 5)
ax_pos.set_ylim(np.nanmin(interp_y) - 5, np.nanmax(interp_y) + 5)
ax_pos.set_title('Mouse Position')

# Ax for head-direction cell
circle = plt.Circle((0, 0), 1, fill=False)
ax_hd.add_patch(circle)
hd_dots = []
ax_hd.set_xlim(-1.2, 1.2)
ax_hd.set_ylim(-1.2, 1.2)
ax_hd.set_xticks([])
ax_hd.set_yticks([])
ax_hd.set_aspect('equal')
ax_hd.set_title('Head-Direction at Cell Spikes')

# Preprocess spikes per frame
spikes_per_frame = [[] for _ in range(len(uniform_time))]
spike_idx = 0
for i, t in enumerate(uniform_time):
    while spike_idx < len(spiketrain) and spiketrain[spike_idx] < t:
        angle = spike_headangles[spike_idx]
        spikes_per_frame[i].append(angle)
        spike_idx += 1

# Update function for animation
def update(frame):
    x = interp_x[frame]
    y = interp_y[frame]
    x_rear = interp_rear_x[frame]
    y_rear = interp_rear_y[frame]

    # Draw line from rear to front
    body_line.set_data([x_rear, x], [y_rear, y])

    # Update position
    mouse_dot.set_data([x], [y])
    rear_dot.set_data([x_rear], [y_rear])

    # Remove previous arrow and draw a new one
    arrow.set_positions((x_rear, y_rear), (x, y))

    # Plot new spikes
    # for dot in hd_dots:
    #     dot.remove()
    hd_dots.clear()
    for a in spikes_per_frame[frame]:
        a += np.pi/2
        dot, = ax_hd.plot(np.cos(a), np.sin(a), 'o', color = 'purple', markersize=20, alpha = 0.05, markeredgewidth=0)
        hd_dots.append(dot)

    return [mouse_dot, rear_dot, body_line] + hd_dots



ani = FuncAnimation(fig, update, frames=len(uniform_time), interval=1000/fps, blit=False)
ani = FuncAnimation(fig, update, 1800, interval=1000/fps, blit=False)
# ani = FuncAnimation(fig, update, frames=10, interval=1000/fps, blit=False)
plt.tight_layout()
# plt.show()
ani.save('head_direction_animation.mp4', writer='ffmpeg', fps=fps, dpi=200)