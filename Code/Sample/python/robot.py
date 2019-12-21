import numpy as np
import pybullet as p
import itertools
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt * 10
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
        # self.Edges = np.array([[0,1],[0,2],[1,2],[1,3],[2,3],[3,4],[2,4],[4,5],[3,5]])
        self.L = self.get_laplacian(self.Edges,self.No_of_Robots,False)
        self.K1 = 5
        self.K2 = 3/self.dt
        self.K3 = 10
        self.OK = 50/self.dt
        self.D = 0 /self.dt
        self.Var = 0.1
        self.E = self.get_Incidence(self.Edges,self.No_of_Robots)
        self.Formation = "None"
        # Vertical Formation
        # self.P_Des = np.array([[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]])/2
        # Horizontal Formation
        # self.P_Des = np.array([[1,0],[1,1],[1,2],[0,0],[0,1],[0,2]])/1.5
        # self.P_Des = self.Form_HRec(0.5)
        # Line Formation
        # self.P_Des = self.Form_line(0.5)
        # Square Formation
        self.P_Des = self.Form_square(1)
        self.Reset_Form()
        self.TargetP = np.array([[1,-0.5],[2,-0.5],[3,0],[2.7,1.5],[2.5,3],[2.5,4]])
        self.Tid = 0
        self.test = True
        self.Obstacles = np.array([[0,0],[0,1],[0,2],[0,-1],[0,-2],[1,2],[2,2],[1,-2],[2,-2],[3,0],[3,-1],[3,1],[3,2],[3,-2]])
        self.Timer = 0
        self.FormStable = False
        self.StartTimer = False
        self.Thr = 0.01
        if (self.id == 2):
            self.OffTimer = 0
        else:
            self.OffTimer = 0
    # region My Functions
    def Reset_Form(self):
        self.Z_Des = np.transpose(self.E) @ self.P_Des
        self.Gdsq = np.square(np.transpose(self.E) @ self.P_Des)  # [x^2,y^2]
        self.Gdsq = (self.Gdsq[:, 0] + self.Gdsq[:, 1]).reshape([-1, 1])
        print(self.id, " ", self.Formation)

    def Form_square(self, Sidelength):
        self.Formation = "square"
        Des = np.array([[0, 0], [0, 1], [0.5, 0], [0.5, 1], [1, 0], [1, 1]])*Sidelength
        return Des

    def Form_line(self, Dist):
        self.Formation = "Line"
        Des = np.array([[0, 0], [0, 3], [0, 1], [0, 4], [0, 2], [0, 5]])*Dist
        return Des

    def Form_HRec(self, Scale):
        self.Formation = "HRec"
        Des = np.array([[1, -1], [0, -1], [1, 0], [0, 0], [1, 1], [0, 1]])*Scale
        return Des

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

    def DistJacobian(self,P, Edges):
        x = P[:, 0]
        y = P[:, 1]
        J = np.zeros([Edges.shape[0], 2 * P.shape[0]])
        #     print(J.shape)
        for e in range(0, Edges.shape[0]):
            J[e, 2 * (Edges[e][0])] = 2 * (x[Edges[e][0]] - x[Edges[e][1]])
            J[e, 2 * (Edges[e][0]) + 1] = 2 * (y[Edges[e][0]] - y[Edges[e][1]])
            J[e, 2 * (Edges[e][1])] = -2 * (x[Edges[e][0]] - x[Edges[e][1]])
            J[e, 2 * (Edges[e][1]) + 1] = -2 * (y[Edges[e][0]] - y[Edges[e][1]])
        return J
    # endregion

    # region In-built Functions
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
    # endregion

    def compute_controller(self):
        # region Input Info
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
        # endregion
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

            TarM = np.zeros([6,2])
            TarM[self.id, :] = self.TargetP[self.Tid,:]-pos[0:2]
            Cdiff = Apos-pos[0:2]
            Cdiff = np.sqrt(np.square(Cdiff[:,0])+np.square(Cdiff[:,1]))
            Cdiff = np.sum(Cdiff)
            Ddiff = self.P_Des-self.P_Des[self.id]
            Ddiff = np.sqrt(np.square(Ddiff[:, 0]) + np.square(Ddiff[:, 1]))
            Ddiff = np.sum(Ddiff)
            Tdiff = np.abs(Ddiff - Cdiff)




            # region Obstacles
            Obc = Apos
            # Obc = self.Obstacles
            # Obc = np.vstack([Obs,pos[0:2]])
            Diff = pos[0:2] - Obc
            for m in range(0, Diff.shape[0]):
                if (np.sqrt(np.square(Diff[m, 0]) + np.square(Diff[m, 1]))) > 0.35:
                    Diff[m, :] = np.array([0, 0])

            DiffY = Diff[:, 1].reshape([1, -1])
            DiffX = Diff[:, 0].reshape([1, -1])
            x_odot = np.sum(np.exp(-np.square(DiffX) / self.Var) * DiffX)
            y_odot = np.sum(np.exp(-np.square(DiffY) / self.Var) * DiffY)

            ObsAv = np.array([x_odot, y_odot])
            # endregion


            NewGd = np.square(np.transpose(self.E) @ Apos)
            NewGd = (NewGd[:, 0] + NewGd[:, 1]).reshape([-1, 1])
            G = self.Gdsq - NewGd
            Rg = self.DistJacobian(Apos, self.Edges)
            p_ddot = np.zeros(([6, 2]))

            if (Tdiff < self.Thr):
                self.StartTimer = True


            if(self.StartTimer):
                self.Timer += 1
                if (self.Timer > 500+self.OffTimer):
                    self.FormStable = True
                    self.StartTimer = False
                    self.Timer = 0

            if(self.Tid > 3 and np.sum(TarM[self.id, 0])<5):
                TarM[self.id, 0] = 5
            if (self.Tid > 3 and np.sum(TarM[self.id, 1]) < 5):
                TarM[self.id, 1] = 5
            if (self.Tid > 3 and np.sum(TarM[self.id, 1]) > -5):
                TarM[self.id, 1] = -5
            if (self.Tid > 3 and np.sum(TarM[self.id, 1]) > -5):
                TarM[self.id, 1] = -5

            if (self.Tid > 3 and np.sum(TarM[self.id, :]) < 0.01):
                self.Tid +=1

            if (self.FormStable):
                # Formation Done
                if self.Tid == 0 and self.Formation == "square":
                    self.P_Des = self.Form_HRec(0.5)
                    self.Reset_Form()
                    # self.Tid += 1
                    # self.FormStable = False
                    print(self.P_Des, self.Formation, " ", self.Tid)
                    # self.K1 = 5
                    # self.K2 = 50
            if (self.Tid < self.TargetP.shape[0]-1 and self.FormStable):
                self.Tid += 1
                if(self.Tid == 1):
                    self.K1 = 2
                    self.K3 = 10
                    self.Thr = 0.001
                if (self.Tid == 2):
                    self.K1 = 20
                    self.K3 = 1
                    self.P_Des = self.Form_HRec(0.5)
                    self.Reset_Form()
                self.FormStable = False
            # Linear Control Law
            p_dot = np.zeros([6,2])
            p_dot = -self.K1 * np.matmul(self.L, Apos) + self.K1 * np.matmul(self.E, self.Z_Des)
            p_dot += self.dt * (self.OK / self.Var) * ObsAv
            # p_dot += self.K3 * TarM
            # Non - linear Control Law
            # p_ddot = self.K2 * (np.transpose(Rg) @ G).reshape([-1, 2])
            # p_dot += p_ddot*self.dt
            if(self.id == 0):
                # print(Tdiff,self.TargetP[self.Tid,:],np.sum(G),self.Tid,self.Timer)
                p_dot = self.K3 * TarM
            if (self.id == 0):
                print(Tdiff,self.TargetP[self.Tid,:],np.sum(G),self.Tid,self.Timer)
                # if(self.Tid == 1):
                    # p_dot += -self.K1 * np.matmul(self.L, Apos) + self.K1 * np.matmul(self.E, self.Z_Des)

            dx = p_dot[self.id, 0]
            dy = p_dot[self.id, 1]

                    # Non - linear Control
                    # p_ddot = self.K2 * (np.transpose(Rg) @ G).reshape([-1, 2])
                    # p_ddot += (self.OK / self.Var) * ObsAv
                    # dx = self.dt * p_ddot[self.id, 0]
                    # dy = self.dt * p_ddot[self.id, 1]
            #else:
                # TarM[self.id, :] = Tdiff
                # # Linear Control
                # p_dot = -self.K1 * np.matmul(self.L, Apos) + self.K1 * np.matmul(self.E, self.Z_Des)
                # p_dot += self.dt * (self.OK / self.Var) * ObsAv
                # p_dot += self.K3 * TarM
                # dx = p_dot[self.id, 0]
                # dy = p_dot[self.id, 1]

                # Non - linear Control
                # p_ddot = self.K2 * (np.transpose(Rg) @ G).reshape([-1, 2])
                # p_ddot += self.K3 * TarM
                # p_ddot += (self.OK / self.Var) * ObsAv
                # dx = self.dt * p_ddot[self.id, 0]
                # dy = self.dt * p_ddot[self.id, 1]

            # region Robot Wheel Control
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
            # endregion
        

    
       
