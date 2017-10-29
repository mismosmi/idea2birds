################################################################################
#                       idea2birds                                             #
#                   Bird simulation in python                                  #
################################################################################
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation
from matplotlib.collections import PathCollection
import time
import argparse


class Flock:
    def __init__(self, **kwargs):

        self.args = parse_kwargs(kwargs)
        n = self.args["n"]

        # randomly distributed starting velocities, might come out pretty much the same as if you just set self.velocity = np.zeros((n,2),dtype=np.float32) but avoids division by zero.
        angles = np.random.uniform(0,2*np.pi,self.args["n"])
        self.velocity = (np.array([np.cos(angles),np.sin(angles)])*self.args["v"]).astype(np.float64).T
        self.position = np.random.rand(n,2).astype(np.float64)*[self.args["width"],self.args["height"]]

        # test boid configuration
        if n<=3:
            self.position = np.array([[self.args["width"]/2+15,self.args["height"]/2],[self.args["width"]/2-15,self.args["height"]/2], [self.args["width"]/2,self.args["height"]/2+15]],dtype=np.float32)
            self.velocity = np.array([[self.args["v"],0],[-self.args["v"],0],[self.args["v"],0]],dtype=np.float32)
            self.position = self.position[0:n,:]
            self.velocity = self.velocity[0:n,:]


        if self.args["angle"] == 0:
            self.angle_view = False
        else:
            self.angle_view = self.args["angle"]/360*np.pi*2


    def get_va(self):
        n = self.args["n"]
        if self.args["v"]:
            vsum = np.sum(self.velocity,axis=0)
            return 1./(n*self.args["v"]) * np.sqrt((vsum*vsum).sum())

    def run(self):
        n = self.args["n"]
        self.distance = self.calc_distance()

        mask = (self.distance < self.args["radius"])

        # check if viewing-angle (theta_velocity) +- angle_view/2 matches distance-vector to neighbors (theta(pos1-pos2))
        if self.angle_view:
            # mask *= np.absolute(np.arctan2(self.dy.T, self.dx.T) - np.arctan2(self.velocity[:,1],self.velocity[:,0])) < self.angle_view
            mask *= np.divide( self.dx.T * self.velocity[:,0] + self.dy.T * self.velocity[:,1], self.distance*self.args["v"], np.ones_like(self.distance), where=self.distance!=0) > np.cos(self.angle_view)


        mask_count = np.maximum(mask.sum(axis=1), 1).reshape(self.args["n"],1)

        self.calc_velocity(mask,mask_count)

        if self.args["eta"]:
            self.random_turn()

        self.position += self.velocity
        self.position %= [self.args["width"], self.args["height"]]


    def calc_distance(self):
        n = self.args["n"]
        """
        Calculate distance between birds. Return n x n array with each row
        corresponding to each bird's distance to the others
        """
        # subtract each element of the position array from the others
        self.dx = np.subtract.outer(self.position[:, 0], self.position[:, 0])
        self.dy = np.subtract.outer(self.position[:, 1], self.position[:, 1])

        # wrap around
        np.subtract(self.dx, np.sign(self.dx)*self.args["width"], out=self.dx, where=np.absolute(self.dx) > self.args["width"]/2)
        np.subtract(self.dy, np.sign(self.dy)*self.args["height"], out=self.dy, where=np.absolute(self.dy) > self.args["height"]/2)

        # return hypotenuse of corresponding elements i.e. distance
        return np.hypot(self.dx, self.dy)

    def calc_velocity(self, mask, count):
        # Compute the average velocity of local neighbours
        target = np.dot(mask, self.velocity)/count

        # Compute steering
        norm = np.sqrt((target*target).sum(axis=1)).reshape(self.args["n"], 1)
        target = self.args["v"] * np.divide(target, norm, out=target, where=norm != 0)

        self.velocity = target
        return

    def random_turn(self):
        angles = np.random.uniform(-self.args["eta"]/2,self.args["eta"]/2,self.args["n"])

        # Option 1 using addition theorem
        c = np.cos(angles)
        s = np.sin(angles)
        self.velocity[:,0] = self.velocity[:,0]*c - self.velocity[:,1]*s
        self.velocity[:,1] = self.velocity[:,1]*c + self.velocity[:,0]*s

        # Option 2 using tan/arctan with same result
        #av = np.arctan2(self.velocity[:,1],self.velocity[:,0])
        #av += angles
        #self.velocity = np.array([np.cos(av),np.sin(av)]).reshape(self.args["n"],2)

        return


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
    global flock, collection, trace, param_record

    # record parameters
    if param_record:
        param_record["va"][args[0]] = flock.get_va()

    # Flock updating
    flock.run()
    # print(flock.get_va())

    collection._scale = 10
    collection._translate = flock.position
    collection._rotate = -np.pi/2 + np.arctan2(flock.velocity[:, 1],
                                               flock.velocity[:, 0])
    collection.update()




defaults = {
    "angle": 0,
    "v": 0.03,
    "width": 5,
    "height": 5,
    "n": 100,
    "radius": 1,
    "frames": 1000,
    "vfile": "birds.mp4",
    "fps": 40,
    "eta": 0,
    "scale": 100
    }


def parse_kwargs(args):
    for key, arg in defaults.items():
        if key not in args:
            args[key] = arg

    if "rho" in args and args["rho"]:
        args["L"] = np.sqrt(args["n"]/args["rho"])

    if "L" in args and args["L"]:
        args["width"] = args["L"]
        args["height"] = args["L"]

    if "scale" in args and args["scale"]:
        args["width"] = int(args["scale"] * args["width"])
        args["height"] = int(args["scale"] * args["height"])
        args["radius"] *= args["scale"]
        args["v"] *= args["scale"]
    else:
        args["width"] = int(args["width"])
        args["height"] = int(args["height"])

    return args


# -----------------------------------------------------------------------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--angle", "-a", help="Boid field of Vision [deg], default="+str(defaults["angle"]), type=float, default=defaults["angle"])
    parser.add_argument("-v", help="Velocity, default="+str(defaults["v"]), type=float, default=defaults["v"])
    parser.add_argument("--width", help="Range for x-coordinate, default="+str(defaults["width"]), type=float, default=defaults["width"])
    parser.add_argument("--height", help="Range for y-coordinate, default="+str(defaults["height"]),type=float, default=defaults["height"])
    parser.add_argument("-L", help="Set side length at once: equals --width L --height L", type=float)
    parser.add_argument("--n", "-n", help="Number of boids, default="+str(defaults["n"]),type=int, default=defaults["n"])
    parser.add_argument("--eta", help="Generates Angles -eta/2 < delta Theta < eta/2, default="+str(defaults["eta"]), type=float, default=defaults["eta"])
    parser.add_argument("--radius", "-r", help="Viewing Radius, default="+str(defaults["radius"]), type=float, default=defaults["radius"])
    parser.add_argument("--export", "-e", help="Export video file and exit", action="store_true")
    parser.add_argument("--frames", help="Number of frames for export to video or parameter record file, default="+str(defaults["frames"]), type=int, default=defaults["frames"])
    parser.add_argument("--vfile", help="Out-File for video export, default="+str(defaults["vfile"]), type=str, default=defaults["vfile"])
    parser.add_argument("--fps", help="Set Video Framerate for playback/export, default="+str(defaults["fps"]), type=int, default=defaults["fps"])
    parser.add_argument("--scale", "-s", help="Scale field size, radius, v, default="+str(defaults["scale"]), type=float, default=defaults["scale"])
    parser.add_argument("--rho", help="Set constant density and calculate L from n and rho", type=float)
    parser.add_argument("--out", "-o", help="Specify output File for recording Parameters, Filetype: .npz", type=str)
    parser.add_argument("--batch", "-b", help="Batch mode: no live display, no video export", action="store_true")



    args = parser.parse_args()
    
    flock = Flock(**vars(args))

    fig = plt.figure(figsize=(10, 10*flock.args["height"]/flock.args["width"]), facecolor="white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], aspect=1, frameon=False)
    collection = MarkerCollection(flock.args["n"])
    ax.add_collection(collection._collection)
    ax.set_xlim(0, flock.args["width"])
    ax.set_ylim(0, flock.args["height"])
    ax.set_xticks([])
    ax.set_yticks([])
    


    if args.out:
        # initialize storage arrays
        param_record = {
            "frames": args.frames,
            "va": np.zeros(args.frames),
            "eta": args.eta,
            "rho": flock.args["n"]/(flock.args["width"]*flock.args["height"]),
            "v": args.v if args.v else [args.min_velocity,args.max_velocity]
            }

        # run loop
        if args.batch:
            for i in range(0,args.frames):
                va[i] = flock.get_va()
                flock.run()
    else:
        param_record = False

    interval = 10 if args.fps==defaults["fps"] else 1000/args.fps

    if not args.batch:
        if args.export:
            animation = FuncAnimation(fig, update, interval=interval, frames=args.frames)
            animation.save(args.vfile, fps=args.fps, dpi=80, bitrate=-1, codec="libx264",
                       extra_args=['-pix_fmt', 'yuv420p'],
                       metadata={'artist': 'Nicolas P. Rougier'})

        else:
            animation = FuncAnimation(fig, update, interval=interval, frames=args.frames, repeat=not args.out)
            plt.show()

    if args.out:
        # save to file
        np.savez(args.out, **param_record)



