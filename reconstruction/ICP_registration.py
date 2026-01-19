import open3d as o3d

def ICP_registration(source, target, threshold,trans_init):
    print("Apply point-to-point ICP")
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        trans_init,o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return icp_result