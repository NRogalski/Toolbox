

import numpy as np
import trimesh
from collections import defaultdict
from Geometry import *


def fun_Normals(Faces,Vertices):

    # Face and vertex normals of the triangle mesh

    # Computation of the edge vectors
    e_0 = Vertices[Faces[:,2]]-Vertices[Faces[:,1]]
    e_1 = Vertices[Faces[:,0]]-Vertices[Faces[:,2]]
    e_2 = Vertices[Faces[:,1]]-Vertices[Faces[:,0]]

    # Computation of the face areas
    Areas = 0.5*np.sqrt(np.sum(np.cross(e_0,e_1,axis=1)**2,axis=1))

    # Computation of the face normals
    Face_Normal = fun_Euclidean_Norm(np.cross(e_0,e_1,axis=1))

    # Computation of the vertex normals (after: https://doi.org/10.1080/10867651.1999.10487501)
    Vertex_Normal = np.array([[0.0]*3]*len(Vertices))
    for i_Face in range(len(Faces)):
        # Computation of the weights
        wfv_0 = Areas[i_Face]/(np.sum(e_1[i_Face]**2)*np.sum(e_2[i_Face]**2))
        wfv_1 = Areas[i_Face]/(np.sum(e_0[i_Face]**2)*np.sum(e_2[i_Face]**2))
        wfv_2 = Areas[i_Face]/(np.sum(e_1[i_Face]**2)*np.sum(e_0[i_Face]**2))
        # Computation of the vertex normals
        Vertex_Normal[Faces[i_Face,0]] = Vertex_Normal[Faces[i_Face,0]]+wfv_0*Face_Normal[i_Face]
        Vertex_Normal[Faces[i_Face,1]] = Vertex_Normal[Faces[i_Face,1]]+wfv_1*Face_Normal[i_Face]
        Vertex_Normal[Faces[i_Face,2]] = Vertex_Normal[Faces[i_Face,2]]+wfv_2*Face_Normal[i_Face]
    Vertex_Normal = fun_Euclidean_Norm(Vertex_Normal)

    return Face_Normal,Vertex_Normal


def fun_Voronoi_Areas(Faces,Vertices):

    # Voronoi areas of the triangle mesh (after: https://doi.org/10.1007/978-3-662-05105-4)

    # Computation of the edge vectors
    e_0 = Vertices[Faces[:,2]]-Vertices[Faces[:,1]]
    e_1 = Vertices[Faces[:,0]]-Vertices[Faces[:,2]]
    e_2 = Vertices[Faces[:,1]]-Vertices[Faces[:,0]]

    # Computation of the face areas
    Areas = 0.5*np.sqrt(np.sum(np.cross(e_0,e_1,axis=1)**2,axis=1))

    # Computation of the Voronoi areas
    # Face_Voronoi: Voronoi areas at each vertex, for each face
    # Vertex_Voronoi: Voronoi areas at each vertex, for the 1-ring face neighborhood
    Face_Voronoi = np.array([[0.0]*3]*len(Faces))
    Vertex_Voronoi = np.array([0.0]*len(Vertices))
    for i_Face in range(len(Faces)):
        # Cotangent at vertex v_0
        Cotangent_0 = fun_Cotangent(Vertices[Faces[i_Face,1]],Vertices[Faces[i_Face,2]],Vertices[Faces[i_Face,0]])
        # Cotangent at vertex v_1
        Cotangent_1 = fun_Cotangent(Vertices[Faces[i_Face,2]],Vertices[Faces[i_Face,0]],Vertices[Faces[i_Face,1]])
        # Cotangent at vertex v_2
        Cotangent_2 = fun_Cotangent(Vertices[Faces[i_Face,0]],Vertices[Faces[i_Face,1]],Vertices[Faces[i_Face,2]])
        # If the triangle is obtus at v_0
        if Cotangent_0<=0:
            Voronoi_Area_0 = Areas[i_Face]/2
            Voronoi_Area_1 = Areas[i_Face]/4
            Voronoi_Area_2 = Areas[i_Face]/4
        # If the triangle is obtus at v_1
        elif Cotangent_1<=0:
            Voronoi_Area_0 = Areas[i_Face]/4
            Voronoi_Area_1 = Areas[i_Face]/2
            Voronoi_Area_2 = Areas[i_Face]/4
        # If the triangle is obtus at v_2
        elif Cotangent_2<=0:
            Voronoi_Area_0 = Areas[i_Face]/4
            Voronoi_Area_1 = Areas[i_Face]/4
            Voronoi_Area_2 = Areas[i_Face]/2
        # If the triangle is non-obtus
        else:
            Voronoi_Area_0 = 1/8*(np.sum(e_0[i_Face]**2)*(Cotangent_1+Cotangent_2))
            Voronoi_Area_1 = 1/8*(np.sum(e_1[i_Face]**2)*(Cotangent_2+Cotangent_0))
            Voronoi_Area_2 = 1/8*(np.sum(e_2[i_Face]**2)*(Cotangent_0+Cotangent_1))
        Face_Voronoi[i_Face] = [Voronoi_Area_0,Voronoi_Area_1,Voronoi_Area_2]
        Vertex_Voronoi[Faces[i_Face,0]] = Vertex_Voronoi[Faces[i_Face,0]]+Face_Voronoi[i_Face,0]
        Vertex_Voronoi[Faces[i_Face,1]] = Vertex_Voronoi[Faces[i_Face,1]]+Face_Voronoi[i_Face,1]
        Vertex_Voronoi[Faces[i_Face,2]] = Vertex_Voronoi[Faces[i_Face,2]]+Face_Voronoi[i_Face,2]

    return Face_Voronoi,Vertex_Voronoi


def fun_Curvatures(Faces,Vertices):

    # Principal curvatures and directions at the vertices of the triangle mesh (after: https://doi.org/10.1109/TDPVT.2004.1335277)

    # Computation of the edge vectors
    e_0 = Vertices[Faces[:,2]]-Vertices[Faces[:,1]]
    e_1 = Vertices[Faces[:,0]]-Vertices[Faces[:,2]]
    e_2 = Vertices[Faces[:,1]]-Vertices[Faces[:,0]]

    # Computation of the mesh normals
    Face_Normal,Vertex_Normal = fun_Normals(Faces,Vertices)

    # Computation of the initial coordinate system associated with each vertex
    up = np.array([[0.0]*3]*len(Vertices))
    for i_Face in range(len(Faces)):
        up[Faces[i_Face]] = e_0[i_Face]
    up = fun_Euclidean_Norm(np.cross(up,Vertex_Normal,axis=1))
    vp = fun_Euclidean_Norm(np.cross(Vertex_Normal,up,axis=1))

    # Computation of the Voronoi area-weights
    Face_Voronoi,Vertex_Voronoi = fun_Voronoi_Areas(Faces,Vertices)
    wfp = Face_Voronoi/Vertex_Voronoi[Faces].reshape(len(Faces),3)

    # Computation of the Second Fundamental Tensor (SFT)
    Face_SFT = [np.zeros((2,2)) for _ in range(Faces.shape[0])]
    Vertex_SFT = [np.zeros((2,2)) for _ in range(Vertices.shape[0])]
    for i_Face in range(len(Faces)):
        # Per-face computation
        # Computation of the coordinate system associated with each face
        uf = fun_Euclidean_Norm(e_0[i_Face])
        vf = fun_Euclidean_Norm(np.cross(Face_Normal[i_Face],uf))
        # Least squares solving of the linear constraints on the elements of the SFT
        n = Vertex_Normal[Faces[i_Face]]
        A = np.array([[np.dot(e_0[i_Face],uf),np.dot(e_0[i_Face],vf),0],[0,np.dot(e_0[i_Face],uf),np.dot(e_0[i_Face],vf)],[np.dot(e_1[i_Face],uf),np.dot(e_1[i_Face],vf),0],[0,np.dot(e_1[i_Face],uf),np.dot(e_1[i_Face],vf)],[np.dot(e_2[i_Face],uf),np.dot(e_2[i_Face],vf),0],[0,np.dot(e_2[i_Face],uf),np.dot(e_2[i_Face],vf)]])
        b = np.array([[np.dot(n[2]-n[1],uf)],[np.dot(n[2]-n[1],vf)],[np.dot(n[0]-n[2],uf)],[np.dot(n[0]-n[2],vf)],[np.dot(n[1]-n[0],uf)],[np.dot(n[1]-n[0],vf)]])
        x = np.linalg.lstsq(A,b)
        Face_SFT[i_Face] = np.array([[x[0][0,0],x[0][1,0]],[x[0][1,0],x[0][2,0]]])
        for i_Vertex in range(3):
            # Computation of the new coordinate system associated with each vertex
            Axis = fun_Euclidean_Norm(np.cross(Vertex_Normal[Faces[i_Face,i_Vertex]],Face_Normal[i_Face]))
            Angle = np.acos(np.dot(Vertex_Normal[Faces[i_Face,i_Vertex]],Face_Normal[i_Face]))
            if Angle==0:
                R = np.eye(3)
            else:
                R = fun_Axang2rotm(Axis,Angle)
            up_Rot = (R@up[Faces[i_Face,i_Vertex]].T).T
            vp_Rot = (R@vp[Faces[i_Face,i_Vertex]].T).T    
            # SFT expressed in the new coordinate system
            ep = np.array([np.dot(up_Rot,uf),np.dot(up_Rot,vf)])@Face_SFT[i_Face]@np.array([np.dot(up_Rot,uf),np.dot(up_Rot,vf)]).T
            ef = np.array([np.dot(up_Rot,uf),np.dot(up_Rot,vf)])@Face_SFT[i_Face]@np.array([np.dot(vp_Rot,uf),np.dot(vp_Rot,vf)]).T
            eg = np.array([np.dot(vp_Rot,uf),np.dot(vp_Rot,vf)])@Face_SFT[i_Face]@np.array([np.dot(vp_Rot,uf),np.dot(vp_Rot,vf)]).T
            # Per-vertex computation
            Vertex_SFT[Faces[i_Face,i_Vertex]] = Vertex_SFT[Faces[i_Face,i_Vertex]]+wfp[i_Face,i_Vertex]*np.array([[ep,ef],[ef,eg]])

    # Computation of the principal curvatures and directions at the vertices
    k_1,k_2 = [np.array([0.0]*len(Vertices)) for _ in range(2)]
    PD_1,PD_2 = [np.array([[0.0]*3]*len(Vertices)) for _ in range(2)]
    for i_Vertex in range(len(Vertices)):
        Eigen_Val,Eigen_Vect = np.linalg.eig(Vertex_SFT[i_Vertex])
        PC,Order = np.sort(Eigen_Val)[::-1],np.argsort(Eigen_Val)[::-1]
        k_1[i_Vertex] = PC[0]
        k_2[i_Vertex] = PC[1]
        PD = np.array([[Eigen_Vect[0,0]*up[i_Vertex]+Eigen_Vect[1,0]*vp[i_Vertex]],[Eigen_Vect[0,1]*up[i_Vertex]+Eigen_Vect[1,1]*vp[i_Vertex]]])
        PD_1[i_Vertex] = PD[Order[0]]
        PD_2[i_Vertex] = PD[Order[1]]

    return k_1,k_2,PD_1,PD_2


def fun_nth_Ring(Faces,Vertices,n):

    # n-th ring of each face of the triangle mesh (edge-sharing)

    # Computation of the adjacent faces (Face_Neighbors[i] returns the faces sharing an edge with the face i)
    Face_Adjacency = trimesh.Trimesh(vertices=Vertices,faces=Faces).face_adjacency
    Face_Neighbors = defaultdict(list)
    for i_Edge in Face_Adjacency:
        Face_Neighbors[i_Edge[0]].append(i_Edge[1])
        Face_Neighbors[i_Edge[1]].append(i_Edge[0])
    # Face_Neighbors are extended by duplicating existing elements when len(Face_Neighbors[i])<=2
    for i_Face in list(Face_Neighbors.keys()):
        Neighbors = Face_Neighbors[i_Face]
        if len(Neighbors)<=2:
            Face_Neighbors[i_Face] = (Neighbors*3)[:3]
    Face_Neighbors = np.array([item[1] for item in sorted(Face_Neighbors.items())])

    # Computation of the n-th ring of each face
    n_Ring = [[] for _ in range(Faces.shape[0])]
    Flag = np.zeros((Faces.shape[0],),dtype=bool)
    for i_Face in range(len(Faces)):
        Neighbors = i_Face
        Flag[i_Face] = True
        for _ in range(n):
            Neighbors = np.unique(Face_Neighbors[Neighbors])
            Neighbors = Neighbors[~Flag[Neighbors]]
            Flag[Neighbors] = True
        n_Ring[i_Face] = np.where(Flag)
        Flag[:] = False

    return n_Ring
