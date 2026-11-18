from PIL import Image
import numpy as np

img = Image.open('pacman/analysis/procyon.png').convert('RGB')
arr = np.array(img)
green = (arr[..., 1] > 200) & (arr[..., 0] == 0) & (arr[..., 2] == 0)

# 8x8 tile map
tile_green = np.zeros((31, 28), dtype=bool)
for r in range(31):
    for c in range(28):
        if np.sum(green[r*8:(r+1)*8, c*8:(c+1)*8]) >= 4:
            tile_green[r, c] = True

# Also start tile
tile_green[23, 13] = True

# Start tracing from (23, 13)
# Follow the green path by moving along tiles
curr = (23, 13)
visited_edges = set()
path = [curr]
dir_prev = None

# Let's inspect the first 50 steps
print('Tracing starting at (23, 13)...')
for step in range(250):
    r, c = curr
    nbrs = []
    # Possible directions: left, down, up, right
    for dr, dc, dn in [(0, -1, 'left'), (1, 0, 'down'), (0, 1, 'right'), (-1, 0, 'up')]:
        nr, nc = r + dr, (c + dc) % 28
        if 0 <= nr < 31 and tile_green[nr, nc]:
            edge = ((r, c), (nr, nc))
            rev_edge = ((nr, nc), (r, c))
            if edge not in visited_edges and rev_edge not in visited_edges:
                nbrs.append((nr, nc, dn, edge))
    
    if not nbrs:
        print(f'End of path at {curr} after {step} steps.')
        break
    
    # If multiple, prefer continuing in same direction, or turning smoothly
    if dir_prev:
        same_dir = [n for n in nbrs if n[2] == dir_prev]
        if same_dir:
            next_tile = same_dir[0]
        else:
            next_tile = nbrs[0]
    else:
        # Initial direction from (23, 13) is LEFT
        next_tile = [n for n in nbrs if n[2] == 'left'][0]
        
    curr = (next_tile[0], next_tile[1])
    dir_prev = next_tile[2]
    visited_edges.add(next_tile[3])
    path.append(curr)

print('Total tiles in path:', len(path))
corners = []
for i in range(1, len(path)-1):
    dr1, dc1 = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
    dr2, dc2 = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
    if (dr1, dc1) != (dr2, dc2):
        corners.append(path[i])

print('Waypoints (corners):', len(corners))
print('Corners:', corners)
