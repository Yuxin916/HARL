from robotarium_python_simulator.rps.utilities.misc import *
from robotarium_gym.scenarios.base import BaseVisualization

class Visualize(BaseVisualization):
    def __init__(self, args):
        self.line_width = 2
        self.CM = plt.cm.get_cmap('cool', 10)
        self.show_figure = True
    
    def initialize_markers(self, robotarium, agents):
        self.marker_sizes = []
        for agent in agents.agents:
            self.marker_sizes.append(determine_marker_size(robotarium, agent.radius))

        self.robot_markers = [ robotarium.axes.scatter( \
            agents.agent_poses[0,ii], agents.agent_poses[1,ii],
            s=self.marker_sizes[ii], marker='o', facecolors='none',\
            edgecolors = (self.CM(int(agents.agents[ii].radius * 12))), linewidth=self.line_width )\
            for ii in range(agents.num_robots) ]
    
    def update_markers(self, robotarium, agents ):
        for i in range(agents.agent_poses.shape[1]):
            self.robot_markers[i].set_offsets(agents.agent_poses[:2,i].T)
            # Next two lines updates the marker sizes if the figure window size is changed.
            self.robot_markers[i].set_sizes([determine_marker_size(robotarium, agents.agents[i].radius)])