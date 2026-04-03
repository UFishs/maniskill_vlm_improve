import h5py

src_path = "/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/base_traj_500.rgb.pd_joint_delta_pos.physx_cpu.h5"
dst_path = "/cephfs/gyshare/ruizihang/maniskill_vlm_improve/demos/StackThree-v1/motionplanning/example.h5"

traj_names = ["traj_0", "traj_1"]

with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
    for traj in traj_names:
        if traj not in src:
            raise KeyError(f"{traj} not found in source file")

        src.copy(traj, dst)

print("Done.")
