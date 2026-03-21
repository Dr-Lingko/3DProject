import open3d as o3d
import numpy as np
import os
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings("ignore")

class PointCloudPlaneSection:
    def __init__(self, pcd_path: str):
        """
        初始化点云截面分析工具（兼容所有Open3D版本 + Python 3.11.9）
        :param pcd_path: 配准完成的点云文件路径（PLY/PCD格式）
        """
        # 1. 校验文件是否存在
        if not os.path.exists(pcd_path):
            raise FileNotFoundError(
                f"\n❌ PLY文件不存在！请检查：\n"
                f"  预期路径：{pcd_path}\n"
                f"  请确认文件是否存在，或路径是否写错"
            )
        
        # 2. 读取点云（兼容所有版本）
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        
        # 3. 校验点云是否为空
        if len(self.pcd.points) == 0:
            raise ValueError(
                f"\n❌ 点云读取为空！可能原因：\n"
                f"  1. 文件损坏：用MeshLab打开检查\n"
                f"  2. 格式错误：确保是PLY/PCD格式\n"
                f"  3. 权限不足：以管理员身份运行"
            )
        
        # 4. 下采样优化（降低交互卡顿）
        self.pcd = self.pcd.voxel_down_sample(voxel_size=0.001)
        self.points_np = np.asarray(self.pcd.points, dtype=np.float64)
        
        # 核心变量
        self.clicked_points = []  # 存储3个选点
        self.plane_eq = None      # 平面方程 [a,b,c,d]
        self.section_area = 0.0   # 横截面积
        self.vis = None           # 可视化对象

    def fit_plane_by_3points(self):
        """三点拟合平面方程 ax + by + cz + d = 0"""
        p1, p2, p3 = self.clicked_points
        # 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2).astype(np.float64)
        normal = normal / np.linalg.norm(normal)  # 归一化
        a, b, c = normal
        d = - (a*p1[0] + b*p1[1] + c*p1[2])
        self.plane_eq = [a, b, c, d]

    def project_point_cloud_to_plane(self):
        """将点云投影到拟合平面"""
        if not self.plane_eq:
            return
        a, b, c, d = self.plane_eq
        n = np.array([a, b, c], dtype=np.float64)
        
        # 向量化投影计算（高效）
        t = (a*self.points_np[:,0] + b*self.points_np[:,1] + c*self.points_np[:,2] + d) / (a**2 + b**2 + c**2)
        self.proj_points = self.points_np - t[:, np.newaxis] * n

    def calculate_section_area(self):
        """计算投影点云的凸包面积（横截面积）"""
        # 将3D投影点转2D坐标
        a, b, c, _ = self.plane_eq
        # 生成平面内的两个正交基
        if abs(c) > 1e-8:
            u = np.array([1, 0, -a/c], dtype=np.float64)
        else:
            u = np.array([1, -a/b, 0], dtype=np.float64)
        u = u / np.linalg.norm(u)
        v = np.cross(np.array([a, b, c]), u)
        v = v / np.linalg.norm(v)
        
        # 2D投影
        xy_points = np.column_stack([
            np.dot(self.proj_points, u),
            np.dot(self.proj_points, v)
        ])
        
        # 拟合凸包计算面积
        try:
            hull = ConvexHull(xy_points)
            self.section_area = hull.volume  # 2D凸包volume即面积
        except:
            raise ValueError("选点共线！请重新选3个不共线的点。")

    def select_points_and_calculate(self):
        """核心交互逻辑：选3个点 → 计算面积"""
        # 创建兼容所有版本的可视化窗口（支持选点）
        self.vis = o3d.visualization.VisualizerWithEditing()
        self.vis.create_window(
            window_name="点云截面分析（Python 3.11.9）",
            width=1200, height=800
        )
        # 添加原始点云（灰色）
        self.pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.vis.add_geometry(self.pcd)
        
        # 打印操作提示（关键！）
        print("="*60)
        print("🎯 点云平面截面分析工具（兼容所有版本）")
        print("操作说明：")
        print("1. 鼠标左键拖拽：旋转视角 | 右键拖拽：平移 | 滚轮：缩放")
        print("2. 按下 Shift + 鼠标左键点击点云：选择点（共需选3个）")
        print("3. 选满3个点后自动计算面积 | 按 Q/ESC 关闭窗口")
        print("="*60)
        
        # 运行可视化（选点模式）
        self.vis.run()
        self.vis.destroy_window()
        
        # 获取选中的点（Open3D原生选点结果）
        picked_indices = self.vis.get_picked_points()
        if len(picked_indices) < 3:
            raise ValueError(f"仅选中 {len(picked_indices)} 个点！需要选3个点。")
        
        # 取前3个选点的坐标
        self.clicked_points = [self.points_np[i] for i in picked_indices[:3]]
        print(f"\n✅ 已选中3个点：")
        for i, p in enumerate(self.clicked_points):
            print(f"  点{i+1}：{np.round(p, 4)}")
        
        # 拟合平面 → 投影点云 → 计算面积
        self.fit_plane_by_3points()
        self.project_point_cloud_to_plane()
        self.calculate_section_area()
        
        # 输出结果
        print("\n" + "="*50)
        print(f"📌 拟合平面方程：{np.round(self.plane_eq[0],4)}x + {np.round(self.plane_eq[1],4)}y + {np.round(self.plane_eq[2],4)}z + {np.round(self.plane_eq[3],4)} = 0")
        print(f"📏 该平面截点云的横截面积：{self.section_area:.6f} 平方单位")
        print("="*50)
        
        # 可视化投影结果（绿色点云）
        proj_pcd = o3d.geometry.PointCloud()
        proj_pcd.points = o3d.utility.Vector3dVector(self.proj_points)
        proj_pcd.paint_uniform_color([0, 1, 0])
        
        # 标记选中的3个点（红色小球）
        spheres = []
        for p in self.clicked_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
            sphere.paint_uniform_color([1, 0, 0])
            sphere.translate(p)
            spheres.append(sphere)
        
        # 显示结果
        o3d.visualization.draw_geometries([self.pcd, proj_pcd] + spheres,
                                          window_name="结果预览（绿色=投影点云，红色=选点）")

if __name__ == "__main__":
    # ==================== 替换为你的点云路径 ====================
    PCD_PATH = r"E:\3DProject\D2\main_object_fixed\super_sqz.ply"

    # 异常捕获（友好提示）
    try:
        analyzer = PointCloudPlaneSection(PCD_PATH)
        analyzer.select_points_and_calculate()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
    except ValueError as e:
        print(f"\n❌ {e}")
    except ImportError:
        print("\n❌ 依赖库缺失！执行：pip install open3d numpy scipy")
    except Exception as e:
        print(f"\n❌ 运行出错：{e}")
        print("💡 解决建议：以管理员身份运行，或升级显卡驱动")