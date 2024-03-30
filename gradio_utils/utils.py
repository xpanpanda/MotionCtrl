import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def vis_camera(RT_list, rescale_T=1): # 含有多个camera pose的列表+缩放引自
    fig = go.Figure()
    showticklabels = True
    visible = True # 可见性
    scene_bounds = 2 # 边界
    base_radius = 2.5 # 基础半径
    zoom_scale = 1.5 # 缩放比例
    fov_deg = 50.0 # 视场角
    
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1), (3, 4)] # 边缘链接
    
    colors = px.colors.qualitative.Plotly # 颜色列表
    
    cone_list = []
    n = len(RT_list)
    for i, RT in enumerate(RT_list):
        R = RT[:,:3]
        T = RT[:,-1]/rescale_T
        cone = calc_cam_cone_pts_3d(R, T, fov_deg) # 计算给定旋转和缩放后的视锥顶点
        cone_list.append((cone, (i*1/n, "green"), f"view_{i}")) # 将视锥顶点、颜色和图例标签添加到 cone_list 中

    
    for (cone, clr, legend) in cone_list:
        for (i, edge) in enumerate(edges):
            # 对于视锥的每条边提取对应的顶点坐标
            (x1, x2) = (cone[edge[0], 0], cone[edge[1], 0])
            (y1, y2) = (cone[edge[0], 1], cone[edge[1], 1])
            (z1, z2) = (cone[edge[0], 2], cone[edge[1], 2])
            # 添加一条线段，表示视锥的边缘，设置颜色和线宽，并根据是否是第一个视锥来决定是否显示图例。
            fig.add_trace(go.Scatter3d(
                x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                line=dict(color=clr, width=3),
                name=legend, showlegend=(i == 0))) 
    # 更新图形的布局，包括设置图形的高度，自动尺寸，边距，图例
    fig.update_layout(
                    height=500,
                    autosize=True,
                    # hovermode=False,
                    margin=go.layout.Margin(l=0, r=0, b=0, t=0),
                    
                    showlegend=True,
                    legend=dict(
                        yanchor='bottom',
                        y=0.01,
                        xanchor='right',
                        x=0.99,
                    ),
                    scene=dict(
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1.0),
                        camera=dict(
                            center=dict(x=0.0, y=0.0, z=0.0),
                            up=dict(x=0.0, y=-1.0, z=0.0),
                            eye=dict(x=scene_bounds/2, y=-scene_bounds/2, z=-scene_bounds/2),
                            ),

                        xaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        yaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        ),
                            
                        
                        zaxis=dict(
                            range=[-scene_bounds, scene_bounds],
                            showticklabels=showticklabels,
                            visible=visible,
                        )
                    ))
    return fig

# 世界坐标系到相机坐标系的变换矩阵
def calc_cam_cone_pts_3d(R_W2C, T_W2C, fov_deg, scale=0.1, set_canonical=False, first_frame_RT=None):
    fov_rad = np.deg2rad(fov_deg) # 视场角度转化为弧度制
    R_W2C_inv = np.linalg.inv(R_W2C) 

    # Camera pose center:
    T = np.zeros_like(T_W2C) - T_W2C #反平移
    T = np.dot(R_W2C_inv, T) # 将相机坐标系转化为世界坐标系
    cam_x = T[0]
    cam_y = T[1]
    cam_z = T[2]
    if set_canonical: 
        # 如果 set_canonical 为True，并且提供了 first_frame_RT，
        # 则将相机位置设置为相对于第一帧相机姿态的坐标形式，再将其转换回世界坐标系
        T = np.zeros_like(T_W2C) # 初始化一个长度为n的0向量
        T = np.dot(first_frame_RT[:,:3], T) + first_frame_RT[:,-1]
        # 我不懂，上面两步明明就可以表示为一步T=first_frame_RT[:,-1]
        T = T - T_W2C 
        T = np.dot(R_W2C_inv, T)
        cam_x = T[0]
        cam_y = T[1]
        cam_z = T[2]

    # vertex
    corn1 = np.array([np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale 
    corn2 = np.array([-np.tan(fov_rad / 2.0), 0.5*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn3 = np.array([0, -0.25*np.tan(fov_rad / 2.0), 1.0]) *scale
    corn4 = np.array([0, -0.5*np.tan(fov_rad / 2.0), 1.0]) *scale

    corn1 = corn1 - T_W2C
    corn2 = corn2 - T_W2C
    corn3 = corn3 - T_W2C
    corn4 = corn4 - T_W2C
    
    corn1 = np.dot(R_W2C_inv, corn1)
    corn2 = np.dot(R_W2C_inv, corn2)
    corn3 = np.dot(R_W2C_inv, corn3) 
    corn4 = np.dot(R_W2C_inv, corn4) 

    # Now attach as offset to actual 3D camera position:
    corn_x1 = corn1[0]
    corn_y1 = corn1[1]
    corn_z1 = corn1[2]
    
    corn_x2 = corn2[0]
    corn_y2 = corn2[1]
    corn_z2 = corn2[2]
    
    corn_x3 = corn3[0]
    corn_y3 = corn3[1]
    corn_z3 = corn3[2]
    
    corn_x4 = corn4[0]
    corn_y4 = corn4[1]
    corn_z4 = corn4[2]
            

    xs = [cam_x, corn_x1, corn_x2, corn_x3, corn_x4, ]
    ys = [cam_y, corn_y1, corn_y2, corn_y3, corn_y4, ]
    zs = [cam_z, corn_z1, corn_z2, corn_z3, corn_z4, ]

    return np.array([xs, ys, zs]).T



    # T_base = [
    #             [1.,0.,0.],             ## W2C  x 的正方向： 相机朝左  left
    #             [-1.,0.,0.],            ## W2C  x 的负方向： 相机朝右  right
    #             [0., 1., 0.],           ## W2C  y 的正方向： 相机朝上  up     
    #             [0.,-1.,0.],            ## W2C  y 的负方向： 相机朝下  down
    #             [0.,0.,1.],             ## W2C  z 的正方向： 相机往前  zoom out
    #             [0.,0.,-1.],            ## W2C  z 的负方向： 相机往前  zoom in
    #         ]   
    # radius = 1
    # n = 16
    # # step = 
    # look_at = np.array([0, 0, 0.8]).reshape(3,1)
    # # look_at = np.array([0, 0, 0.2]).reshape(3,1)

    # T_list = []
    # base_R = np.array([[1., 0., 0.],
    #                 [0., 1., 0.],
    #                 [0., 0., 1.]])
    # res = [] 
    # res_forsave = []
    # T_range = 1.8



    # for i in range(0, 16):
    #     # theta = (1)*np.pi*i/n

    #     R = base_R[:,:3]
    #     T = np.array([0.,0.,1.]).reshape(3,1) * (i/n)*2
    #     RT = np.concatenate([R,T], axis=1)
    #     res.append(RT)
        
    # fig = vis_camera(res)
    
