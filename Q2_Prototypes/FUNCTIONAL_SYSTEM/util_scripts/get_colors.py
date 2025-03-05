import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors

# Get the dictionary of accepted color names and hex values
colors = mcolors.cnames
sorted_colors = sorted(colors.items())

# Determine grid dimensions (here, we'll use 8 columns)
cols = 8
rows = len(sorted_colors) // cols + (len(sorted_colors) % cols > 0)

fig, ax = plt.subplots(figsize=(cols * 1.5, rows * 1.5))
ax.set_xlim(0, cols)
ax.set_ylim(0, rows)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Accepted Colors in Matplotlib", fontsize=20)

for i, (name, hex_value) in enumerate(sorted_colors):
    col = i % cols
    row = rows - 1 - (i // cols)
    
    # Create a rectangle patch filled with the color
    rect = patches.Rectangle((col, row), 1, 1, facecolor=hex_value)
    ax.add_patch(rect)
    
    # Calculate brightness to decide text color for readability
    rgb = mcolors.to_rgb(hex_value)
    brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    text_color = 'black' if brightness > 0.5 else 'white'
    
    ax.text(col + 0.5, row + 0.5, name, ha='center', va='center', fontsize=8, color=text_color)

plt.show()
