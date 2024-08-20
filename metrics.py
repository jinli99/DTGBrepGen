from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import topods_Shell, topods_Wire, topods_Face
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_SHELL, TopAbs_FACE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeFix import ShapeFix_Wire
from multiprocessing.pool import Pool
from tqdm import tqdm
import trimesh
import os


def check_brep_validity(step_file_path):
    # Read the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        print("错误：无法读取STEP文件")
        return

    step_reader.TransferRoot()
    shape = step_reader.Shape()

    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True

    # Check all wires
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods_Face(face_explorer.Current())
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods_Wire(wire_explorer.Current())

            # Create a ShapeFix_Wire object
            wire_fixer = ShapeFix_Wire(wire, face, 0.01)
            wire_fixer.Load(wire)
            wire_fixer.SetFace(face)
            wire_fixer.SetPrecision(0.01)
            wire_fixer.SetMaxTolerance(1)
            wire_fixer.SetMinTolerance(0.0001)

            # Fix the wire
            wire_fixer.Perform()
            fixed_wire = wire_fixer.Wire()

            # Analyze the fixed wire
            wire_analysis = ShapeAnalysis_Wire(fixed_wire, face, 0.01)
            wire_analysis.Load(fixed_wire)
            wire_analysis.SetPrecision(0.01)
            wire_analysis.SetSurface(BRep_Tool.Surface(face))

            # 1. Check wire edge order
            order_status = wire_analysis.CheckOrder()
            if order_status != 0:  # 0 means no error
                print(f"Wire order issue detected: {order_status}")
                wire_order_ok = False

            # 2. Check wire self-intersection
            if wire_analysis.CheckSelfIntersection():
                wire_self_intersection_ok = False

            wire_explorer.Next()
        face_explorer.Next()

    # 3. Check for bad edges in shells
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = topods_Shell(shell_explorer.Current())
        shell_analysis = ShapeAnalysis_Shell()
        shell_analysis.LoadShells(shell)

        if shell_analysis.HasBadEdges():
            shell_bad_edges_ok = False

        shell_explorer.Next()

    # Output results
    print(f"线圈边缘顺序正确: {'是' if wire_order_ok else '否'}")
    print(f"线圈无自交: {'是' if wire_self_intersection_ok else '否'}")
    print(f"壳体无坏边: {'是' if shell_bad_edges_ok else '否'}")

    return wire_order_ok and wire_self_intersection_ok and shell_bad_edges_ok


def check_mesh(mesh_path):
    mesh = trimesh.load(mesh_path)

    is_watertight = mesh.is_watertight

    return 1 if is_watertight else 0


def load_data_with_prefix(root_folder, prefix):
    data_files = []

    # Walk through the directory tree starting from the root folder
    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            # Check if the file ends with the specified prefix
            if filename.endswith(prefix):
                file_path = os.path.join(root, filename)
                data_files.append(file_path)

    return data_files


def main():
    path = "samples/test_eg/eval/test_eg_8layers"
    files = load_data_with_prefix(path, '.stl')
    convert_iter = Pool(os.cpu_count()).imap(check_mesh, files)
    valid = 0
    for status in tqdm(convert_iter, total=len(files)):
        valid += status
    print(valid, len(files), valid/len(files))

    with open(os.path.join(path, 'results.txt'), "w") as file:
        file.write('Total: ' + str(len(files)) + ', Valid: ' + str(valid) + ', Ratio: ' + str(valid/len(files)) + '\n')


if __name__ == '__main__':
    # main()
    print(check_mesh("samples/test_eg/eval/test_eg_6layers/Z31H65VgqcFmPew_0.stl"))
