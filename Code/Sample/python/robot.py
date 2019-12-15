import numpy as np
import pybullet as p
import itertools


class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between body and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []

        self.No_of_Robots = 6
        self.Edges = self.Complete_Graph(self.No_of_Robots)
        self.L = self.get_laplacian(self.Edges,self.No_of_Robots,False)
        self.K1 = 1
        self.K2 = 1
        self.K3 = 3
        self.E = self.get_Incidence(self.Edges,self.No_of_Robots)
        # Vertical Formation
        self.P_Des = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]])
        # Horizontal Formation
        # self.P_Des = np.array([[0,0],[1,0],[0,1],[1,1],[0,2],[1,2]])
        self.Z_Des = np.transpose(self.E)@self.P_Des
        self.test = True

    def get_Incidence(self, Edges, n_vertices):
        E = np.zeros([n_vertices, Edges.shape[0]])
        for x in range(Edges.shape[0]):
            E[Edges[x, 0], x] = -1
            E[Edges[x, 1], x] = 1
        return E

    def Complete_Graph(self, n_vertices):
        Edges = np.zeros([0, 2])
        for i in range(0, n_vertices):
            for j in range(i + 1, n_vertices):
                EdgesTemp = [i, j]
                Edges = np.vstack([Edges, EdgesTemp])
        Edges = Edges.astype(int)
        return Edges

    # Get Laplacian Function
    def get_laplacian(self, Edges, n_vertices, Directed):
        A = np.zeros([n_vertices, n_vertices])
        D = np.zeros([n_vertices, n_vertices])
        for x in Edges:
            if Directed:
                A[x[1], x[0]] = 1
                D[x[1], x[1]] += 1
            else:
                A[x[0], x[1]] = 1
                A[x[1], x[0]] = 1
                D[x[0], x[0]] += 1
                D[x[1], x[1]] += 1

        L = D - A
        return L

    def reset(self):
        """
        Moves the robot back to its initial position 
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        """
        returns a list of received messages, each element of the list is a tuple (a,b)
        where a= id of the sending robot and b= message (can be any object, list, etc chosen by user)
        Note that the message will only be received if the robot is a neighbor (i.e. is close enough)
        """
        return self.messages_received
        
    def send_message(self, robot_id, message):
        """
        sends a message to robot with id number robot_id, the message can be any object, list, etc
        """
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        """
        returns a list of neighbors (i.e. robots within 2m distance) to which messages can be sent
        """
        return self.neighbors
    
    def compute_controller(self):
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """
        
        # here we implement an example for a consensus algorithm
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()
        
        #send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, pos)
        
        # check if we received the position of our neighbors and compute desired change in position
        # as a function of the neighbors (message is composed of [neighbors id, position])
        dx = 0.
        dy = 0.
        # print(messages)
        if messages:
            # similar to laplacian but for each robot
            # for m in messages:
            #     dx += m[1][0] - pos[0]
            #     dy += m[1][1] - pos[1]

            # position of All robots
            Apos = np.zeros([6,2])
            Apos[self.id,:]=pos[0:2]
            for m in messages:
                Apos[m[0],:]=m[1][0:2]

            # # if self.id  == 0:
            # #     print(self.L[self.id]@Apos)

            p_dot = -self.K1 * np.matmul(self.L, Apos) + self.K1 * np.matmul(self.E, self.Z_Des) + self.K3 *

            # if self.test:
            #     print(dx,dy)
            dx = p_dot[self.id,0]
            dy = p_dot[self.id,1]
            # if self.test:
            #     print(dx,dy)
            #     self.test = False
            # integrate
            des_pos_x = pos[0] + self.dt * dx
            des_pos_y = pos[1] + self.dt * dy
        
            #compute velocity change for the wheels
            vel_norm = np.linalg.norm([dx, dy]) #norm of desired velocity
            if vel_norm < 0.01:
                vel_norm = 0.01
            des_theta = np.arctan2(dy/vel_norm, dx/vel_norm)
            right_wheel = np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            left_wheel = -np.sin(des_theta-rot)*vel_norm + np.cos(des_theta-rot)*vel_norm
            self.set_wheel_velocity([left_wheel, right_wheel])
        

    
       
