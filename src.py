import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataclasses import dataclass
width, height = 20, 20
color1 = (255, 0, 0) 
color2 = (0, 0, 255) 

# Create a NumPy array for the image
pixel_data = np.zeros((height, width, 3), dtype=np.uint8)

# Fill the array with a checkerboard pattern
pixel_data[::2, ::2] = color1  # Fill even rows and columns with color1
pixel_data[1::2, 1::2] = color1  # Fill odd rows and columns with color1
pixel_data[::2, 1::2] = color2  # Fill even rows, odd columns with color2
pixel_data[1::2, ::2] = color2  # Fill odd rows, even columns with color2

# Create an image from the array
image = Image.fromarray(pixel_data)
# Save and show the image
image.save("image.png")

@dataclass 
class solution: 
    x: np.ndarray
    energies: np.ndarray
    accepts: np.ndarray
    rejects: np.ndarray
    configs: list[np.ndarray]
    """ 
    Container class for the solution and statistics 
    :param x: The final pixel grid of data points 
    :param energies: The energy series of the graph 
    :param accepts: The cumalative number of accepts 
    :param rejects: The cumalative number of rejects
    :param configs: Accepted configs
    """
@dataclass
class Annealer: 
    ml: int 
    iterations: int 
    T_0: float 
    T_f: float 
    """
    Container class for the annealer hyperparameters 
    
    :param ml: The markov chain length 
    :param iterations: The number of iterations 
    :param T_0: The initial temperature 
    :param T_f: The final temperature  
    """
    
    def optimize(self, x: np.ndarray)->solution: 
        """
        :param x: The initial pixel grid 
        :returns solution: The solution object 
        :raises ValueError: If the input grid is not a square
        """
        
        # Check to make sure the pixel grid is a square 
        if(x.shape[0] != x.shape[1] or x.shape[2] != 3): raise ValueError("The pixel grid is not of appropriate dimensionality.")
        
        # Initializations 
        energies = np.zeros(self.iterations)
        accepts = np.zeros(self.iterations)
        rejects = np.zeros(self.iterations)
        temperatures = np.linspace(self.T_0, self.T_f, self.iterations)
        configs = []
        # Nearest Neighbors
        right_neighbor = np.zeros_like(x)
        left_neighbor = np.zeros_like(x)
        bottom_neighbor = np.zeros_like(x)
        top_neighbor = np.zeros_like(x)
        
        right_neighbor[:, :-1] = x[:, 1:]
        right_neighbor[:, -1] = x[:, -1]
        
        left_neighbor[:, 1:] = x[:, :-1]
        left_neighbor[:, 0] = x[:, 0]
        
        top_neighbor[1:, :] = x[:-1, :]
        top_neighbor[0, :] = x[0, :]
        
        bottom_neighbor[:-1, :] = x[1:, :]
        bottom_neighbor[-1, :] = x[-1, :]
        energy = 1/(x.shape[0]**2)*np.sum(np.linalg.norm(x-right_neighbor, axis=-1) + np.linalg.norm(x-left_neighbor, axis=-1) + np.linalg.norm(x-top_neighbor, axis=-1)  + np.linalg.norm(x-bottom_neighbor, axis=-1))  
        optimal_x = x
        for iter in range(self.iterations): 
            # Log current values 
            energies[iter] = energy
            
            for i in range(self.ml):
                # Switch two elements of the grid 
                new_grid = np.copy(x)
                ran_i1 = np.random.randint(new_grid.shape[0])
                ran_j1 = np.random.randint(new_grid.shape[0])
                ran_i2 = np.random.randint(new_grid.shape[0])
                ran_j2 = np.random.randint(new_grid.shape[0])
                
                pixel = np.copy(new_grid[ran_i1, ran_j1])
                new_grid[ran_i1, ran_j1] = new_grid[ran_i2, ran_j2]
                new_grid[ran_i2, ran_j2] = pixel
                
                right_neighbor[:, :-1] = new_grid[:, 1:]
                left_neighbor[:, 1:] = new_grid[:, :-1]
                top_neighbor[1:, :] = new_grid[:-1, :]
                bottom_neighbor[:-1, :] = new_grid[1:, :]
                
                
                right_neighbor[:, -1] = new_grid[:, -1]
                left_neighbor[:, 0] = new_grid[:, 0]
                top_neighbor[0, :] = new_grid[0, :]
                bottom_neighbor[-1, :] = new_grid[-1, :]
                new_energy = 1/(x.shape[0]**2)*np.sum(np.linalg.norm(x-right_neighbor, axis=-1) + np.linalg.norm(x-left_neighbor, axis=-1) + np.linalg.norm(x-top_neighbor, axis=-1)  + np.linalg.norm(x-bottom_neighbor, axis=-1))  
                diff = new_energy-energy
                br =  np.exp(-diff/temperatures[iter]) 
                if(diff < 0 or  br > np.random.uniform()):
                    x = new_grid 
                    configs.append(x)
                    energy = new_energy
                    accepts[iter] = 1
                else: 
                    rejects[iter] = 1
                if(diff < 0): 
                    optimal_x = new_grid
        
    
    
        solargs = { 
            'x': optimal_x, 
            'energies': energies, 
            'accepts': np.cumsum(accepts), 
            'rejects': np.cumsum(rejects),
            'configs': configs
        }
        return solution(**solargs)

solver = Annealer(iterations=1000, ml=8000, T_0=1000, T_f=1)
y = solver.optimize(pixel_data)
image = Image.fromarray(y.x)
image.save("final.png")
plt.close()
plt.plot(np.arange(y.energies.size), y.energies)
plt.savefig('energies.png', dpi=800, bbox_inches='tight')
plt.close()
plt.plot(np.arange(y.accepts.size), y.accepts)
plt.savefig('accepts.png', dpi=800, bbox_inches='tight')
plt.close() 
plt.plot(np.arange(y.rejects.size), y.rejects)
plt.savefig('rejects.png', dpi=800, bbox_inches='tight')
plt.close()
plt.imshow(y.x, interpolation='nearest')
plt.show()
plt.close()
# for i in range(len(y.configs)): 
#         plt.imshow(y.configs[i], interpolation='nearest')
#         plt.savefig(f'frames/img_{i:09d}')
#         plt.close()
# import os
# import cv2
# # Path to the folder containing images
# image_folder = 'frames'
# output_video = 'output_video.mp4'
# frame_rate = 60  # Frames per second

# # Get list of all image files in the folder, sorted in order
# images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
# print(images)
# # Read the first image to get the frame size
# first_image = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = first_image.shape

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
# video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

# # Loop through images and write them to the video
# for image in images:
#     img_path = os.path.join(image_folder, image)
#     frame = cv2.imread(img_path)
#     video.write(frame)

# # Release the VideoWriter object
# video.release()
# print("Video created successfully!")