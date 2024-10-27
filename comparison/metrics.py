import trimesh
import os
from multiprocessing.pool import Pool
from tqdm import tqdm
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopoDS import topods_Shell, topods_Wire, topods_Face, topods_Edge
from OCC.Core.ShapeAnalysis import ShapeAnalysis_Wire, ShapeAnalysis_Shell, ShapeAnalysis_FreeBounds
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_WIRE, TopAbs_SHELL, TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.ShapeFix import ShapeFix_Wire
from utils import load_data_with_prefix


def check_brep_validity(step_file_path):
    # Read the STEP file
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(step_file_path)

    if status != IFSelect_RetDone:
        print("Error: Unable to read STEP file")
        return

    step_reader.TransferRoot()
    shape = step_reader.Shape()

    # Initialize check results
    wire_order_ok = True
    wire_self_intersection_ok = True
    shell_bad_edges_ok = True
    brep_closed_ok = True  # Initialize closed BRep check

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
                # print(f"Wire order issue detected: {order_status}")
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

    # 4. Check if BRep is closed (no free edges)
    free_bounds = ShapeAnalysis_FreeBounds(shape)
    free_edges = free_bounds.GetOpenWires()
    edge_explorer = TopExp_Explorer(free_edges, TopAbs_EDGE)
    num_free_edges = 0
    while edge_explorer.More():
        edge = topods_Edge(edge_explorer.Current())
        num_free_edges += 1
        # print(f"Free edge: {edge}")
        edge_explorer.Next()
    if num_free_edges > 0:
        brep_closed_ok = False

    # Output results
    # print(f"Wire edge order correct: {'Yes' if wire_order_ok else 'No'}")
    # print(f"Wire has no self-intersections: {'Yes' if wire_self_intersection_ok else 'No'}")
    # print(f"Shell has no bad edges: {'Yes' if shell_bad_edges_ok else 'No'}")
    # print(f"BRep is closed: {'Yes' if brep_closed_ok else 'No'}")

    return int(wire_order_ok and wire_self_intersection_ok and shell_bad_edges_ok and brep_closed_ok)


def compute_novelty(source_path, target_path):
    pass


def compute_uniqueness(source_path, target_path):
    pass


def check_mesh_validity(mesh_path):
    mesh = trimesh.load(mesh_path)

    is_watertight = mesh.is_watertight

    return 1 if is_watertight else 0


class ValidMetric:
    """
    A class to compute validity metrics for BREP and mesh files.
    """

    def __init__(self, input_path, save_path=None):
        self.brep_files = load_data_with_prefix(input_path, '.step')
        self.mesh_files = load_data_with_prefix(input_path, '.stl')
        self.save_path = save_path or input_path
        self.bad_brep = []
        self.bad_mesh = []

    @staticmethod
    def process_file(file):
        if file.endswith('.step'):
            status = check_brep_validity(file)
        elif file.endswith('.stl'):
            status = check_mesh_validity(file)
        else:
            raise ValueError(f"Unsupported file type: {file}")
        return file, status

    def write_results(self, file_type, total, valid):
        with open(os.path.join(self.save_path, 'results.txt'), "a") as file:
            file.write(f'{file_type} Check {{Total: {total}}}, {{Valid: {valid}}}, {{Ratio: {valid/total:.4f}}}\n')

    def compute_metric(self, parallel=True):
        for file_type, files in [("Brep", self.brep_files), ("Mesh", self.mesh_files)]:
            valid = 0
            bad_files = []

            if len(files) == 0:
                continue

            if parallel:
                with Pool(os.cpu_count()) as pool:
                    results = list(tqdm(pool.imap(self.process_file, files), total=len(files)))
                for file, status in results:
                    valid += status
                    if not status:
                        bad_files.append(file)
            else:
                for file in tqdm(files):
                    _, status = self.process_file(file)
                    valid += status
                    if not status:
                        bad_files.append(file)

            print(f"{file_type}: {valid}/{len(files)} = {valid/len(files):.4f}")
            self.write_results(file_type, len(files), valid)

            if file_type == "Brep":
                self.bad_brep = bad_files
            else:
                self.bad_mesh = bad_files


def main():

    xx = ValidMetric(input_path='samples/deepcad')
    xx.compute_metric(parallel=True)


if __name__ == '__main__':
    main()
