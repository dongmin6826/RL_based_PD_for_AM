from config import * 
import mesh_processor as mp  
import PD_interface

def extract_features(mesh):
    # (1) Volume
    part_volume = mesh.volume

    # (2) Size of the bounding box
    bounding_box = mesh.bounding_box.extents

    # (3) Concavity
    concavity = part_volume - mesh.convex_hull.volume

    # (4) Support volume
    Utility = PD_interface.Utility()
    mesh = [mesh]
    _, sup_vol = Utility.orientation(Utility.create_obj(mesh))
    #print(sup_vol)

    return part_volume, concavity, bounding_box, sup_vol


def create_env():
    # Import the initial model
    processor = mp.MeshProcessor()
    processor.load_mesh(MESH_PATH)

    part_volume, concavity, bounding_box, sup_vol = extract_features(processor.mesh)
    '''
    print("Initial model ==== ")
    print("Validation (Watertight): ", processor.mesh.is_watertight)
    print("Volume: ", part_volume)
    print("concavity: ", concavity)
    print("bounding_box: ", bounding_box)
    print("support_volume: ", support_volume)
    '''
    # processor.pyvista_visualize(processor.mesh)

    # Create a PD tree
    PD_tree = {1: {"Vol": part_volume, "BB-X": bounding_box[0], "BB-Y": bounding_box[1], "BB-Z": bounding_box[2],
                   "Conc": concavity, "SupVol": sup_vol[0], "Mesh": processor.mesh}}

    # Create a list of decomposed parts
    part_list = [1]

    return PD_tree, part_list


# def cap_current_state(PD_tree, decomposed_parts):
#     # Capture the current state of the PD environment
#     # BUILD ORIENTATION DETERMINATION

#     return state


def deter_build_orientation(trimesh_model):   
    Utility = PD_interface.Utility()
    build_orientation, sup_vol = Utility.orientation(
        Utility.create_obj(trimesh_model))

    # SET THE CURRENT BUILD ORIENTATION TO THE DEFAULT

    return build_orientation, sup_vol


def decompose_parts(ACTION, part_list, PD_tree):
    Utility=PD_interface.Utility()
    MeshProcessor=mp.MeshProcessor()
    Part=part_list[round(ACTION[0])] 

    # ACTION[0] : PART ID of the part to be decomposed
    # ACTION[1] : CUTTING PLANE COORDINATE & ANGLE 
    start_point = [ACTION[1],ACTION[2],ACTION[3]]
    plain_normal=[ACTION[4],ACTION[5],ACTION[6]]

    meshes,check = MeshProcessor.trimesh_cut(PD_tree[Part]['Mesh'],start_point, plain_normal)

    obj,sup_vol=deter_build_orientation(meshes)

    #Cal Reward
    meshes=Utility.create_trimesh(obj)
 
    if len(meshes) > 0 and check==True:
        i = 1
        for mesh in meshes:
            part_volume, concavity, bounding_box, sup_vol = extract_features(mesh)
            
            #print("Mesh{} Validation (Watertight): ".format(i), mesh.is_watertight)
            #print("Mesh{} Volume: ".format(i), part_volume)
            # mp.processor.pyvista_visualize(mesh)
            PartID = Part*10+i
            i += 1

            # Update the PD tree
            PD_tree[PartID] = {"Vol": part_volume, "BB-X":bounding_box.extents[0],"BB-Y": bounding_box.extents[1],"BB-Z": bounding_box.extents[2],
                               "Conc": concavity, "SupVol":sup_vol[i-2],"Mesh":mesh}

            # Update the list of decomposed parts
            part_list.append(PartID)
        part_list.remove(Part)
  
    reward=0
    for part in part_list:
        reward=reward+PD_tree[part]["SupVol"]
        print(f"{part}:{PD_tree[part]["SupVol"]}")

    print("=================")
    print("Sum of SupVol:",reward)
    print("=================")

    return PD_tree, part_list,reward


def cal_reward(min_volume_of_surrport_struct):
    # Calculate the reward based on the current state
    
    return -min_volume_of_surrport_struct

'''
PD_tree, part_list=create_env()
PD_tree, part_list,reward=decompose_parts([1,0,0,0,-20,-20,100],part_list,PD_tree)
PD_tree, part_list,reward=decompose_parts([12,0,0,0,-200,-20,100],part_list,PD_tree)
'''