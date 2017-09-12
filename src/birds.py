################################################################################
#                       idea2birds                                             #
#                   Bird simulation in python                                  #
################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection

class Flock:
    def __init__(self, n=500, width=640, height=380):
        self.velocity = np.zeros((n,2),dtype=np.float32)
        self.position = np.zeros((n,2),dtype=np.float32)

        
        

    def run(self):
        pass

    def calc_distance(self):
        pass

    def calc_alignment(self):
        pass

    def calc_cohesion(self, mask, count):
        # Compute the gravity center of local neighbours
        center = np.dot(mask, self.position)/count.reshape(n, 1)
        
        # Compute direction toward the center
        target = center - self.position
        
        # Normalize the result
        norm = np.sqrt((target*target).sum(axis=1)).reshape(n, 1)
        target *= np.divide(target, norm, out=target, where=norm != 0)
        
        # Cohesion at constant speed (max_velocity)
        target *= max_velocity
        
        # Compute the resulting steering
        self.cohesion = target - velocity

    def calc_separation(self):
        # Compute the repulsion force from local neighbours
        repulsion = np.dstack((self.dx, self.dy))
        
        # Force is inversely proportional to the distance
        repulsion = np.divide(repulsion, distance.reshape(self.n, self.n, 1)**2, out=repulsion,
                              where=distance.reshape(self.n, self.n, 1) != 0)
        
        # Compute direction away from others
        target = (repulsion*mask.reshape(self.n, self.n, 1)).sum(axis=1)/count.reshape(self.n, 1)
        
        # Normalize the result
        norm = np.sqrt((target*target).sum(axis=1)).reshape(self.n, 1)
        target *= np.divide(target, norm, out=target, where=norm != 0)
        
        # Separation at constant speed (max_velocity)
        target *= max_velocity
        
        # Compute the resulting steering
        self.separation = target - self.velocity


class MarkerCollection:
    """
    Marker collection
    """

    def __init__(self, n=100):
        v = np.array([(-0.25, -0.25), (+0.0, +0.5), (+0.25, -0.25), (0, 0)])
        c = np.array([Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
        self._base_vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
        self._vertices = np.tile(v.reshape(-1), n).reshape(n, len(v), 2)
        self._codes = np.tile(c.reshape(-1), n)

        self._scale = np.ones(n)
        self._translate = np.zeros((n, 2))
        self._rotate = np.zeros(n)

        self._path = Path(vertices=self._vertices.reshape(n*len(v), 2),
                          codes=self._codes)
        self._collection = PathCollection([self._path], linewidth=0.5,
                                          facecolor="k", edgecolor="w")

    def update(self):
        n = len(self._base_vertices)
        self._vertices[...] = self._base_vertices * self._scale
        cos_rotate, sin_rotate = np.cos(self._rotate), np.sin(self._rotate)
        R = np.empty((n, 2, 2))
        R[:, 0, 0] = cos_rotate
        R[:, 1, 0] = sin_rotate
        R[:, 0, 1] = -sin_rotate
        R[:, 1, 1] = cos_rotate
        self._vertices[...] = np.einsum('ijk,ilk->ijl', self._vertices, R)
        self._vertices += self._translate.reshape(n, 1, 2)



def update(*args):
    global flock, collection, trace

    # Flock updating
    flock.run()
    collection._scale = 10
    collection._translate = flock.position
    collection._rotate = -np.pi/2 + np.arctan2(flock.velocity[:, 1],
                                               flock.velocity[:, 0])
    collection.update()

    # Trace updating
    if trace is not None:
        P = flock.position.astype(int)
        trace[height-1-P[:, 1], P[:, 0]] = .75
        trace *= .99
        im.set_array(trace)


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    n = 500
    width, height = 640, 360
    flock = Flock(n)
    fig = plt.figure(figsize=(10, 10*height/width), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect=1, frameon=False)
    collection = MarkerCollection(n)
    ax.add_collection(collection._collection)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xticks([])
    ax.set_yticks([])

    # Trace
    trace = None
    if 0:
        trace = np.zeros((height, width))
        im = ax.imshow(trace, extent=[0, width, 0, height], vmin=0, vmax=1,
                       interpolation="nearest", cmap=plt.cm.gray_r)

    animation = FuncAnimation(fig, update, interval=10, frames=1000)
    animation.save('boid.mp4', fps=40, dpi=80, bitrate=-1, codec="libx264",
                   extra_args=['-pix_fmt', 'yuv420p'],
                   metadata={'artist': 'Nicolas P. Rougier'})
    plt.show()

