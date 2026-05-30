import numpy as np

def canonicalize_keypoints(kpts_xyc: np.ndarray, bbox_xyxy: np.ndarray | None = None) -> np.ndarray:
    out = kpts_xyc.copy().astype(np.float32)

    if bbox_xyxy is not None and np.asarray(bbox_xyxy).size == 4:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox_xyxy).reshape(-1)[:4]]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
    else:
        min_xy = out[:, :2].min(axis=0)
        max_xy = out[:, :2].max(axis=0)
        cx = 0.5 * float(min_xy[0] + max_xy[0])
        cy = 0.5 * float(min_xy[1] + max_xy[1])
        w = max(1e-6, float(max_xy[0] - min_xy[0]))
        h = max(1e-6, float(max_xy[1] - min_xy[1]))

    scale = max(w, h)
    out[:, 0] = (out[:, 0] - cx) / scale
    out[:, 1] = (out[:, 1] - cy) / scale
    return out


def _build_features(kpts_xyc: np.ndarray, feature_type: str) -> np.ndarray:
    kpts_xyc = canonicalize_keypoints(kpts_xyc)
    x_raw = kpts_xyc.flatten()

    min_pt = kpts_xyc[:, :2].min(axis=0)
    max_pt = kpts_xyc[:, :2].max(axis=0)
    width = max_pt[0] - min_pt[0]
    height = max_pt[1] - min_pt[1]
    aspect_ratio = width / (height + 1e-6)
    bbox_area = width * height

    spine_vec = kpts_xyc[0, :2] - kpts_xyc[16, :2]

    L_F_SHO = 5
    R_F_SHO = 8
    L_B_HIP = 11
    R_B_HIP = 14
    L_F_FT = 7
    L_B_FT = 13

    hip_dist = np.linalg.norm(kpts_xyc[L_B_HIP, :2] - kpts_xyc[R_B_HIP, :2])
    sho_dist = np.linalg.norm(kpts_xyc[L_F_SHO, :2] - kpts_xyc[R_F_SHO, :2])
    stance_dist = np.linalg.norm(kpts_xyc[L_F_FT, :2] - kpts_xyc[L_B_FT, :2])
    direction_sign = kpts_xyc[L_F_SHO, 0] - kpts_xyc[R_F_SHO, 0]

    torso_pts = kpts_xyc[[L_F_SHO, R_F_SHO, R_B_HIP, L_B_HIP], :2]
    x_coords = torso_pts[:, 0]
    y_coords = torso_pts[:, 1]
    torso_area = 0.5 * np.abs(
        np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1))
    )

    if feature_type == "basic":
        final_features = x_raw
    elif feature_type == "geo":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area]])
    elif feature_type == "full":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec])
    elif feature_type == "foreshortening":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [hip_dist, sho_dist]])
    elif feature_type == "stance":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [stance_dist]])
    elif feature_type == "symmetry":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [direction_sign]])
    elif feature_type == "torso":
        final_features = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [torso_area]])
    elif feature_type == "expert":
        final_features = np.concatenate(
            [x_raw, [aspect_ratio, bbox_area], spine_vec, [hip_dist, sho_dist, stance_dist, direction_sign, torso_area]]
        )
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return final_features.astype(np.float32)
