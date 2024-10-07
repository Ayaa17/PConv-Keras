import os
import itertools
import matplotlib
import matplotlib.pyplot as plt

# Import function that generates random masks
from libs.util import MaskGenerator

# Instantiate mask generator
mask_generator = MaskGenerator(512, 512, 3, rand_seed=42)

# Plot the results
_, axes = plt.subplots(5, 5, figsize=(20, 20))
axes = list(itertools.chain.from_iterable(axes))

for i in range(len(axes)):
    # Generate image
    img = mask_generator.sample()
    print(img)
    # Plot image on axis
    axes[i].imshow(img * 255)

plt.show()