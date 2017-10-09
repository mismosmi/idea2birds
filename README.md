# idea2birds
Simulation of the collective motion of a flock of birds for the course idea to result at university of copenhagen in winter semester 2017/2018

## Usage:
```
usage: birds.py [-h] [--angle ANGLE] [-v V] [--width WIDTH] [--height HEIGHT]
                [-L L] [--n N] [--eta ETA] [--radius RADIUS] [--export]
                [--frames FRAMES] [--vfile VFILE] [--fps FPS] [--scale SCALE]
                [--rho RHO] [--out OUT] [--batch]

optional arguments:
  -h, --help            show this help message and exit
  --angle ANGLE, -a ANGLE
                        Boid field of Vision [deg], default=0
  -v V                  Velocity, default=0.03
  --width WIDTH         Range for x-coordinate, default=5
  --height HEIGHT       Range for y-coordinate, default=5
  -L L                  Set side length at once: equals --width L --height L
  --n N, -n N           Number of boids, default=100
  --eta ETA             Generates Angles -eta/2 < delta Theta < eta/2,
                        default=0
  --radius RADIUS, -r RADIUS
                        Viewing Radius, default=1
  --export, -e          Export video file and exit
  --frames FRAMES       Number of frames for export to video or parameter
                        record file, default=1000
  --vfile VFILE         Out-File for video export, default=birds.mp4
  --fps FPS             Set Video Framerate for export, default=40
  --scale SCALE, -s SCALE
                        Scale field size, radius, v, default=100
  --rho RHO             Set constant density and calculate L from n and rho
  --out OUT, -o OUT     Specify output File for recording Parameters,
                        Filetype: .npz
  --batch, -b           Batch mode: no live display, no video export
```
