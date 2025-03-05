import os
import numpy as np
import yaml
import torch
import torch.multiprocessing as mp
from argparse import Namespace
from tqdm import tqdm
from diffusers import DDPMScheduler, PNDMScheduler
from model import FaceGeomTransformer, EdgeGeomTransformer, VertGeomTransformer, FaceBboxTransformer, FaceEdgeModel, EdgeVertModel
from topology.topoGenerate import SeqGenerator
from topology.transfer import faceVert_from_edgeVert, face_vert_trans, fef_from_faceEdge
from utils import xe_mask, pad_zero, sort_bbox_multi, generate_random_string, make_mask, pad_and_stack, calculate_y
from OCC.Extend.DataExchange import write_step_file
from visualization import *
from inference.brepBuild import (Brep2Mesh, sample_bspline_curve, sample_bspline_surface, create_bspline_curve,
                                 create_bspline_surface, joint_optimize, construct_brep)
from topology.transfer import *
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

text2int = {'uncond':0,
            'bathtub':1,
            'bed':2,
            'bench':3,
            'bookshelf':4,
            'cabinet':5,
            'chair':6,
            'couch':7,
            'lamp':8,
            'sofa':9,
            'table':10
            }

int2text = {v: k for k, v in text2int.items()}


def get_topology(batch, faceEdge_model, edgeVert_model, device, labels, point_data):
    """****************Brep Topology****************"""
    edgeVert_adj = []
    edgeFace_adj = []
    faceEdge_adj = []
    vertFace_adj = []
    fef_adj = []

    valid = 0

    if labels is not None:
        class_label = torch.LongTensor(labels).to(device).reshape(-1, 1)
    else:
        class_label = None
    fail_idx = []

    with torch.no_grad():

        adj_batch = faceEdge_model.sample(num_samples=batch, class_label=class_label, point_data=point_data)               # b*nf*nf

        for i in tqdm(range(batch)):
            adj = adj_batch[i]                                                                     # nf*nf
            non_zero_mask = torch.any(adj != 0, dim=1)
            adj = adj[non_zero_mask][:, non_zero_mask]                                             # nf*nf
            edge_counts = torch.sum(adj, dim=1)                                                    # nf
            if edge_counts.max() > edgeVert_model.max_edge:
                fail_idx.append(i)
                continue
            sorted_ids = torch.argsort(edge_counts)                                                # nf
            adj = adj[sorted_ids][:, sorted_ids]
            edge_indices = torch.triu(adj, diagonal=1).nonzero(as_tuple=False)
            num_edges = adj[edge_indices[:, 0], edge_indices[:, 1]]
            ef_adj = edge_indices.repeat_interleave(num_edges, dim=0)                              # ne*2
            share_id = calculate_y(ef_adj)
            point_data_item = None
            if point_data is not None:
                point_data_item = point_data[i]
                point_data_item = point_data_item.unsqueeze(0)

            # save encoder information
            edgeVert_model.save_cache(edgeFace_adj=ef_adj.unsqueeze(0),
                                      edge_mask=torch.ones((1, ef_adj.shape[0]), device=device, dtype=torch.bool),
                                      share_id=share_id,
                                      class_label=class_label[[i]] if class_label is not None else None,
                                      point_data=point_data_item)
            for try_time in range(10):
                generator = SeqGenerator(ef_adj.cpu().numpy())
                if generator.generate(edgeVert_model, class_label[[i]] if class_label is not None else None, point_data_item):
                    # print("Construct Brep Topology succeed at time %d!" % try_time)
                    edgeVert_model.clear_cache()
                    valid += 1
                    break
            else:
                # print("Construct Brep Topology Failed!")
                edgeVert_model.clear_cache()
                fail_idx.append(i)
                continue

            ev_adj = generator.edgeVert_adj
            fe_adj = generator.faceEdge_adj

            edgeVert_adj.append(torch.from_numpy(ev_adj).to(device))
            edgeFace_adj.append(ef_adj)
            faceEdge_adj.append(fe_adj)
            fv_adj = faceVert_from_edgeVert(fe_adj, ev_adj)
            vf_adj = face_vert_trans(faceVert_adj=fv_adj)
            vertFace_adj.append(vf_adj)
            fef = fef_from_faceEdge(edgeFace_adj=ef_adj.cpu().numpy())
            fef_adj.append(torch.from_numpy(fef).to(device))

    if labels is not None:
        labels = [labels[i] for i in range(len(class_label)) if i not in fail_idx]

    if point_data is not None:
        point_data = [point_data[i] for i in range(len(point_data)) if i not in fail_idx]

    print("topo success:%d"%valid)
    print("topo false:%d"%len(fail_idx))

    return {"edgeVert_adj": edgeVert_adj, "edgeFace_adj": edgeFace_adj, "faceEdge_adj": faceEdge_adj,
            "vertFace_adj": vertFace_adj, "fef_adj": fef_adj, "class_label": labels, "point_data": point_data, "fail_idx": fail_idx}


def get_faceBbox(fef_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Args:
        fef_adj: List of tensors, where each tensor has shape [nf, nf], representing face-edge-face topology.
        model: vertGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
        point_data: [2000, 3]
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(fef_adj)
    device = fef_adj[0].device

    nf = max([i.shape[0] for i in fef_adj])
    fef_adj = [pad_zero(i, max_len=nf, dim=1) for i in fef_adj]
    face_mask = torch.stack([i[1] for i in fef_adj])                  # b*nf
    fef_adj = torch.stack([i[0] for i in fef_adj])                    # b*nf*nf

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        face_bbox = torch.randn((b, nf, 6), device=device)                                 # b*nf*6
        face_bbox, e = xe_mask(x=face_bbox, e=fef_adj.unsqueeze(-1), node_mask=face_mask)  # b*nf*6, b*nf*nf*1
        e = e.squeeze(-1)                                                                  # b*nf*nf
        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             None,
                             point_data,
                             timesteps.expand(face_bbox.shape[0], 1))
            else:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             None,
                             None,
                             timesteps.expand(face_bbox.shape[0], 1))
            face_bbox = pndm_scheduler.step(pred, t, face_bbox).prev_sample
            face_bbox, _ = xe_mask(x=face_bbox, node_mask=face_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*face_bbox.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             None,
                             point_data,
                             timesteps.expand(face_bbox.shape[0], 1))
            else:
                pred = model(face_bbox,
                             e,
                             face_mask,
                             None,
                             None,
                             timesteps.expand(face_bbox.shape[0], 1))
            face_bbox = ddpm_scheduler.step(pred, t, face_bbox).prev_sample
            face_bbox, _ = xe_mask(x=face_bbox, node_mask=face_mask)

    return face_bbox, face_mask


def get_vertGeom(face_bbox, vertFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding boxes.
        vertFace_adj: List of List[List[int]]
        edgeVert_adj: List tensors, where each tensor has shape [ne, 2]
        model: vertGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
        point_data: [2000, 3]
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(face_bbox)
    device = face_bbox[0].device
    vert_per_num = [len(i) for i in vertFace_adj]
    nv = max(vert_per_num)
    vv_adj = torch.zeros((b, nv, nv), device=device, dtype=torch.int)
    for i in range(b):
        indices = edgeVert_adj[i]                              # ne*2
        vv_adj[i, indices[:, 0], indices[:, 1]] = 1
        vv_adj[i, indices[:, 1], indices[:, 0]] = 1            # nv*nv
    vert_mask = make_mask(torch.tensor(vert_per_num, device=device).unsqueeze(-1), nv)   # b*nv

    vertFace_bbox = []
    vertFace_mask = []
    vf = max([max([len(j) for j in i]) for i in vertFace_adj])
    for i in range(b):
        temp1 = [face_bbox[i][j] for j in vertFace_adj[i]]      # [vf*6, ...]
        temp1, temp2 = pad_and_stack(temp1, max_n=vf)           # nv*vf*6, nv*vf
        vertFace_bbox.append(temp1)
        vertFace_mask.append(temp2)
    vertFace_bbox, _ = pad_and_stack(vertFace_bbox)             # b*nv*vf*6
    vertFace_mask, _ = pad_and_stack(vertFace_mask)             # b*nv*vf
    assert vertFace_mask.shape == (b, nv, vf) and vertFace_bbox.shape == (b, nv, vf, 6)

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        vert_geom = torch.randn((b, nv, 3), device=device)      # b*nv*3
        vert_geom, e = xe_mask(x=vert_geom, e=vv_adj.unsqueeze(-1), node_mask=vert_mask)  # b*nv*3, b*nv*nv*1
        e = e.squeeze(-1)                                                                 # b*nv*nv
        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             None,
                             point_data,
                             timesteps.expand(vert_geom.shape[0], 1))
            else:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             None,
                             None,
                             timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = pndm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([vert_geom, vert_geom], dim=0),
                             torch.cat([e, e], dim=0),
                             torch.cat([vert_mask, vert_mask], dim=0),
                             torch.cat([vertFace_bbox, vertFace_bbox], dim=0),
                             torch.cat([vertFace_mask, vertFace_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*vert_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             None,
                             point_data,
                             timesteps.expand(vert_geom.shape[0], 1))
            else:
                pred = model(vert_geom,
                             e,
                             vert_mask,
                             vertFace_bbox,
                             vertFace_mask,
                             None,
                             None,
                             timesteps.expand(vert_geom.shape[0], 1))
            vert_geom = ddpm_scheduler.step(pred, t, vert_geom).prev_sample
            vert_geom, _ = xe_mask(x=vert_geom, node_mask=vert_mask)

    return vert_geom, vert_mask


def get_edgeGeom(face_bbox, vert_geom, edgeFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding box.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology.
        model: edgeGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
        point_data: [2000, 3]
    Returns:
        edge_geom (torch.Tensor): [b, ne, 2] tensor, sampled edge geometry
        edge_mask (torch.Tensor): [b, ne] tensor, edge mask of the sampled edge geometry
    """

    b = len(vert_geom)
    device = vert_geom[0].device
    edgeVert_geom = [vert_geom[i][edgeVert_adj[i]] for i in range(b)]      # [ne*2*3, ...]
    edgeFace_bbox = [face_bbox[i][edgeFace_adj[i]] for i in range(b)]                           # [ne*2*6, ...]

    edgeFace_bbox, edge_mask = pad_and_stack(edgeFace_bbox)                # b*ne*2*6, b*ne
    edgeVert_geom, _ = pad_and_stack(edgeVert_geom)                        # b*ne*2*3
    edgeFace_adj, _ = pad_and_stack(edgeFace_adj)                          # b*ne*2

    ne = edge_mask.shape[1]

    w = 0.6
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        edge_geom = torch.randn((b, ne, 12), device=device)  # b*ne*12

        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             None,
                             point_data,
                             timesteps.expand(edge_geom.shape[0], 1))
            else:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             None,
                             None,
                             timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = pndm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([edge_geom, edge_geom], dim=0),
                             torch.cat([edgeFace_bbox, edgeFace_bbox], dim=0),
                             torch.cat([edgeVert_geom, edgeVert_geom], dim=0),
                             torch.cat([edge_mask, edge_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*edge_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             None,
                             point_data,
                             timesteps.expand(edge_geom.shape[0], 1))
            else:
                pred = model(edge_geom,
                             edgeFace_bbox,
                             edgeVert_geom,
                             edge_mask,
                             None,
                             None,
                             timesteps.expand(edge_geom.shape[0], 1))
            edge_geom = ddpm_scheduler.step(pred, t, edge_geom).prev_sample
            edge_geom, _ = xe_mask(x=edge_geom, node_mask=edge_mask)

    return edge_geom, edge_mask


def get_faceGeom(face_bbox, vert_geom, edge_geom, faceEdge_adj, edgeFace_adj, edgeVert_adj, model, pndm_scheduler, ddpm_scheduler, class_label, point_data):
    """
    Args:
        face_bbox: List of tensors, where each tensor has shape [nf, 6], representing face bounding box.
        vert_geom: List of tensors, where each tensor has shape [nv, 3], representing vertex geometries.
        edge_geom: List of tensors, where each tensor has shape [ne, 12], representing edge geometries.
        faceEdge_adj:
        edgeFace_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-face topology.
        edgeVert_adj: List of tensors, where each tensor has shape [ne, 2], representing edge-vert topology.
        model: faceGeom denoising model
        pndm_scheduler: pndm sampling
        ddpm_scheduler: ddpm sampling
        class_label: List with length b
        point_data: [2000, 3]
    Returns:
        face_geom (torch.Tensor): [b, nf, 48] tensor, sampled face geometry
        face_mask (torch.Tensor): [b, nf, 48] tensor, face mask of the sampled face geometry
    """

    b = len(face_bbox)
    device = face_bbox[0].device

    faceEdge_geom = []
    faceEdge_mask = []
    faceVert_geom = []
    faceVert_mask = []
    fe = 0
    fv = 0
    for idx in range(b):
        faceEdge_geom.append([edge_geom[idx][i] for i in faceEdge_adj[idx]])
        fe = max(fe, max([len(j) for j in faceEdge_geom[-1]]))
        faceVert_geom.append([vert_geom[idx][torch.unique((edgeVert_adj[idx][i]).flatten())] for i in faceEdge_adj[idx]])
        fv = max(fv, max([len(j) for j in faceVert_geom[-1]]))
    faceEdge_geom = [pad_and_stack(i, max_n=fe) for i in faceEdge_geom]
    faceVert_geom = [pad_and_stack(i, max_n=fv) for i in faceVert_geom]
    faceEdge_mask = [i[1] for i in faceEdge_geom]
    faceEdge_geom = [i[0] for i in faceEdge_geom]
    faceVert_mask = [i[1] for i in faceVert_geom]
    faceVert_geom = [i[0] for i in faceVert_geom]
    faceEdge_geom, face_mask = pad_and_stack(faceEdge_geom)      # b*nf*fe*12, b*nf
    faceEdge_mask, _ = pad_and_stack(faceEdge_mask)              # b*nf*fe,
    faceVert_geom, _ = pad_and_stack(faceVert_geom)              # b*nf*fv*3, b*nf
    faceVert_mask, _ = pad_and_stack(faceVert_mask)              # b*nf*fv,
    face_bbox, _ = pad_and_stack(face_bbox)                      # b*nf*6

    w = 0.6
    nf = face_bbox.shape[1]
    if class_label is not None:
        class_label = torch.LongTensor(class_label+[text2int['uncond']]*b).to(device).reshape(-1,1)

    with torch.no_grad():
        face_geom = torch.randn((b, nf, 48), device=device)  # b*ne*48

        pndm_scheduler.set_timesteps(200)
        for t in pndm_scheduler.timesteps[:158]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([face_geom, face_geom], dim=0),
                             torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([faceVert_geom, faceVert_geom], dim=0),
                             torch.cat([faceEdge_geom, faceEdge_geom], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             torch.cat([faceVert_mask, faceVert_mask], dim=0),
                             torch.cat([faceEdge_mask, faceEdge_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*face_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([face_geom, face_geom], dim=0),
                             torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([faceVert_geom, faceVert_geom], dim=0),
                             torch.cat([faceEdge_geom, faceEdge_geom], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             torch.cat([faceVert_mask, faceVert_mask], dim=0),
                             torch.cat([faceEdge_mask, faceEdge_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*face_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(face_geom,
                             face_bbox,
                             faceVert_geom,
                             faceEdge_geom,
                             face_mask,
                             faceVert_mask,
                             faceEdge_mask,
                             None,
                             point_data,
                             timesteps.expand(face_geom.shape[0], 1))
            else:
                pred = model(face_geom,
                             face_bbox,
                             faceVert_geom,
                             faceEdge_geom,
                             face_mask,
                             faceVert_mask,
                             faceEdge_mask,
                             None,
                             None,
                             timesteps.expand(face_geom.shape[0], 1))
            face_geom = pndm_scheduler.step(pred, t, face_geom).prev_sample
            face_geom, _ = xe_mask(x=face_geom, node_mask=face_mask)

        ddpm_scheduler.set_timesteps(1000)
        for t in ddpm_scheduler.timesteps[-250:]:
            timesteps = t.reshape(-1).cuda()
            if class_label is not None and point_data is not None:
                pred = model(torch.cat([face_geom, face_geom], dim=0),
                             torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([faceVert_geom, faceVert_geom], dim=0),
                             torch.cat([faceEdge_geom, faceEdge_geom], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             torch.cat([faceVert_mask, faceVert_mask], dim=0),
                             torch.cat([faceEdge_mask, faceEdge_mask], dim=0),
                             class_label,
                             torch.cat([point_data, point_data], dim=0),
                             timesteps.expand(2*face_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif class_label is not None:
                pred = model(torch.cat([face_geom, face_geom], dim=0),
                             torch.cat([face_bbox, face_bbox], dim=0),
                             torch.cat([faceVert_geom, faceVert_geom], dim=0),
                             torch.cat([faceEdge_geom, faceEdge_geom], dim=0),
                             torch.cat([face_mask, face_mask], dim=0),
                             torch.cat([faceVert_mask, faceVert_mask], dim=0),
                             torch.cat([faceEdge_mask, faceEdge_mask], dim=0),
                             class_label,
                             None,
                             timesteps.expand(2*face_geom.shape[0], 1))
                pred = pred[:b] * (1 + w) - pred[b:] * w
            elif point_data is not None:
                pred = model(face_geom,
                             face_bbox,
                             faceVert_geom,
                             faceEdge_geom,
                             face_mask,
                             faceVert_mask,
                             faceEdge_mask,
                             None,
                             point_data,
                             timesteps.expand(face_geom.shape[0], 1))
            else:
                pred = model(face_geom,
                             face_bbox,
                             faceVert_geom,
                             faceEdge_geom,
                             face_mask,
                             faceVert_mask,
                             faceEdge_mask,
                             None,
                             None,
                             timesteps.expand(face_geom.shape[0], 1))
            face_geom = ddpm_scheduler.step(pred, t, face_geom).prev_sample
            face_geom, _ = xe_mask(x=face_geom, node_mask=face_mask)
    return face_geom, face_mask


def get_brep(args, save_name=None):
    # nf*48, nf*6, nv*3, ne*12, nf*48, ne*2, ne*2, List[List[int]]
    face_bbox, vert_geom, edge_geom, face_geom, edgeFace_adj, edgeVert_adj, faceEdge_adj = args

    assert set(range(face_bbox.shape[0])) == set(np.unique(edgeFace_adj.reshape(-1)).tolist())
    assert set(range(vert_geom.shape[0])) == set(np.unique(edgeVert_adj.reshape(-1)).tolist())

    edge_ncs = []
    for ctrs in edge_geom:
        pcd = sample_bspline_curve(create_bspline_curve(ctrs.reshape(4, 3).astype(np.float64)))  # 32*3
        edge_ncs.append(pcd)
    edge_ncs = np.stack(edge_ncs)  # ne*32*3

    face_ncs = []
    for ctrs in face_geom:
        pcd = sample_bspline_surface(create_bspline_surface(ctrs.reshape(16, 3).astype(np.float64)))  # 32*32*3
        face_ncs.append(pcd)
    face_ncs = np.stack(face_ncs)

    # joint_optimize
    edge_wcs, face_wcs = joint_optimize(edge_ncs, face_ncs, face_bbox, vert_geom, edgeVert_adj, faceEdge_adj, len(edge_ncs), len(face_bbox))

    try:
        solid = construct_brep(face_wcs, edge_wcs, faceEdge_adj, edgeVert_adj)
        if solid is None:
            print('B-rep rebuild failed...')
            return False
        # if save_name is not None:
        #     draw_edge(edge_wcs, save_name=save_name+'.html', auto_open=False)
    except Exception as e:
        print('B-rep rebuild failed...', e)
        return False

    return solid


def process_single_item(data_tuple, save_folder, bbox_scaled):
    j, face_bbox_j, vert_geom_j, edge_geom_j, face_geom_j, edgeFace_adj_j, edgeVert_adj_j, faceEdge_adj_j, class_label_j, point_data_j, file_name_j = data_tuple

    try:
        if class_label_j is not None:
            save_name = f'{save_folder}/{int2text[class_label_j]}_{generate_random_string(10)}'
        elif point_data_j is not None:
            step_file_name = file_name_j
            save_name = f"{save_folder}/{step_file_name}"
        else:
            save_name = f'{save_folder}/{generate_random_string(15)}'

        solid = get_brep((face_bbox_j / bbox_scaled,
                          vert_geom_j / bbox_scaled,
                          edge_geom_j / bbox_scaled,
                          face_geom_j / bbox_scaled,
                          edgeFace_adj_j,
                          edgeVert_adj_j,
                          faceEdge_adj_j), save_name=save_name)

        if solid is False:
            return False

        # np.save(os.path.join(save_name + 'edgeFace.npy'), edgeFace_adj_j)
        # np.save(os.path.join(save_name + 'edgeVert.npy'), edgeVert_adj_j)
        write_step_file(solid, save_name + '.step')
        return True

    except Exception as e:
        print(f"Error processing item {j}: {str(e)}")
        return False


def process_batch(face_bbox, vert_geom, edge_geom, face_geom, edgeFace_adj, edgeVert_adj,
                  faceEdge_adj, args, class_label=None, point_data = None, file_name = None):
    b = len(face_bbox)

    data_list = []
    for j in range(b):
        data_tuple = (
            j,
            face_bbox[j].cpu().numpy(),
            vert_geom[j].cpu().numpy(),
            edge_geom[j].cpu().numpy(),
            face_geom[j].cpu().numpy(),
            edgeFace_adj[j].cpu().numpy(),
            edgeVert_adj[j].cpu().numpy(),
            faceEdge_adj[j],
            class_label[j] if class_label is not None else None,
            point_data if point_data is not None else None,
            file_name[j] if file_name is not None else None
        )
        data_list.append(data_tuple)

    if args.parallel:
        n_processes = min(4, b)
        process_func = partial(process_single_item,
                               save_folder=args.save_folder,
                               bbox_scaled=args.bbox_scaled)

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            results = list(executor.map(process_func, data_list))

    else:
        results = []
        for data_tuple in data_list:
            result = process_single_item(data_tuple,
                                         args.save_folder,
                                         args.bbox_scaled)
            results.append(result)

    success_count = sum(1 for r in results if r)
    return success_count


def main(args):

    # Make project directory if not exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initial Topology Model
    faceEdge_model = FaceEdgeModel(nf=args.max_face,
                                   d_model=args.FaceEdgeModel['d_model'],
                                   nhead=args.FaceEdgeModel['nhead'],
                                   n_layers=args.FaceEdgeModel['n_layers'],
                                   num_categories=args.edge_classes,
                                   use_cf=args.use_cf,
                                   use_pc=args.use_pc)
    faceEdge_model.load_state_dict(torch.load(args.faceEdge_path), strict=False)
    faceEdge_model = faceEdge_model.to(device).eval()
    edgeVert_model = EdgeVertModel(max_num_edge=args.max_num_edge_topo,
                                   max_seq_length=args.max_seq_length,
                                   edge_classes=args.edge_classes,
                                   max_face=args.max_face,
                                   max_edge=args.max_edge,
                                   d_model=args.EdgeVertModel['d_model'],
                                   n_layers=args.EdgeVertModel['n_layers'],
                                   use_cf=args.use_cf,
                                   use_pc=args.use_pc)
    edgeVert_model.load_state_dict(torch.load(args.edgeVert_path), strict=False)
    edgeVert_model = edgeVert_model.to(device).eval()

    # Initial FaceBboxTransformer
    FaceBbox_model = FaceBboxTransformer(n_layers=args.FaceBboxModel['n_layers'],
                                         hidden_mlp_dims=args.FaceBboxModel['hidden_mlp_dims'],
                                         hidden_dims=args.FaceBboxModel['hidden_dims'],
                                         edge_classes=args.edge_classes,
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.use_cf,
                                         use_pc=args.use_pc)
    FaceBbox_model.load_state_dict(torch.load(args.faceBbox_path))
    FaceBbox_model = FaceBbox_model.to(device).eval()

    # Initial VertGeomTransformer
    vertGeom_model = VertGeomTransformer(n_layers=args.VertGeomModel['n_layers'],
                                         hidden_mlp_dims=args.VertGeomModel['hidden_mlp_dims'],
                                         hidden_dims=args.VertGeomModel['hidden_dims'],
                                         act_fn_in=torch.nn.ReLU(),
                                         act_fn_out=torch.nn.ReLU(),
                                         use_cf=args.use_cf,
                                         use_pc=args.use_pc)
    vertGeom_model.load_state_dict(torch.load(args.vertGeom_path))
    vertGeom_model = vertGeom_model.to(device).eval()

    # Initial EdgeGeomTransformer
    edgeGeom_model = EdgeGeomTransformer(n_layers=args.EdgeGeomModel['n_layers'],
                                         edge_geom_dim=args.EdgeGeomModel['edge_geom_dim'],
                                         d_model=args.EdgeGeomModel['d_model'],
                                         nhead=args.EdgeGeomModel['nhead'],
                                         use_cf=args.use_cf,
                                         use_pc=args.use_pc)
    edgeGeom_model.load_state_dict(torch.load(args.edgeGeom_path))
    edgeGeom_model = edgeGeom_model.to(device).eval()

    # Initial FaceGeomTransformer
    faceGeom_model = FaceGeomTransformer(n_layers=args.FaceGeomModel['n_layers'],
                                            face_geom_dim=args.FaceGeomModel['face_geom_dim'],
                                            d_model=args.FaceGeomModel['d_model'],
                                            nhead=args.FaceGeomModel['nhead'],
                                            use_cf=args.use_cf,
                                            use_pc=args.use_pc)
    faceGeom_model.load_state_dict(torch.load(args.faceGeom_path))
    faceGeom_model = faceGeom_model.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start=0.0001,
        beta_end=0.02,
        clip_sample=True,
        clip_sample_range=3
    )

    point = []
    file_names = []
    batch_point_data = None
    batch_file_name = None
    if args.use_pc:
        with open(args.test_path, 'rb') as f:
            batch_file = pickle.load(f)
        for file in batch_file:
            with open(os.path.join('data_process/GeomDatasets', name+'_parsed', file), 'rb') as tf:
                data = pickle.load(tf)
            if 'pc' in data:
                point.append(torch.from_numpy(data['pc']).to(device))
            file_names.append(file[:-4].replace('/', '_'))
    total_sample = 16
    b_each = 16 if args.name == 'furniture' else 32
    class_label = None
    for i in tqdm(range(0, total_sample, b_each)):

        b = min(total_sample, i+b_each)-i
        if args.use_cf:
            class_label = torch.randint(6, 7, (b,)).tolist()
        if args.use_pc:
            batch_point_data = point[i:i + b]
            batch_file_name = file_names[i:i + b]

        # =======================================Brep Topology=================================================== #
        datas = get_topology(b, faceEdge_model, edgeVert_model, device, class_label, batch_point_data)
        fef_adj, edgeVert_adj, faceEdge_adj, edgeFace_adj, vertFace_adj = (datas["fef_adj"],
                                                                           datas["edgeVert_adj"],
                                                                           datas['faceEdge_adj'],
                                                                           datas["edgeFace_adj"],
                                                                           datas["vertFace_adj"])
        class_label = datas['class_label']
        point_data = datas['point_data']
        fail_idx = datas['fail_idx']
        b = len(fef_adj)
        file_name = None
        if batch_file_name is not None:
            file_name = [batch_file_name[j] for j in range(len(batch_file_name)) if j not in fail_idx]

        # ========================================Face Bbox====================================================== #
        face_bbox, face_mask = get_faceBbox(fef_adj, FaceBbox_model, pndm_scheduler, ddpm_scheduler, class_label, point_data)
        face_bbox = [k[l] for k, l in zip(face_bbox, face_mask)]

        face_bbox = [sort_bbox_multi(j) for j in face_bbox]                      # [nf*6, ...]

        # =======================================Vert Geometry=================================================== #
        vert_geom, vert_mask = get_vertGeom(face_bbox, vertFace_adj,
                                            edgeVert_adj, vertGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label, point_data)
        vert_geom = [k[l] for k, l in zip(vert_geom, vert_mask)]

        # =======================================Edge Geometry=================================================== #
        edge_geom, edge_mask = get_edgeGeom(face_bbox, vert_geom,
                                            edgeFace_adj, edgeVert_adj, edgeGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label, point_data)
        edge_geom = [k[l] for k, l in zip(edge_geom, edge_mask)]                 # [ne*12, ...]

        # =======================================Face Geometry=================================================== #
        face_geom, face_mask = get_faceGeom(face_bbox, vert_geom, edge_geom, faceEdge_adj,
                                            edgeFace_adj, edgeVert_adj, faceGeom_model,
                                            pndm_scheduler, ddpm_scheduler, class_label, point_data)
        face_geom = [k[l] for k, l in zip(face_geom, face_mask)]                 # [ne*12, ...]

        # =======================================Construct Brep================================================== #
        success_count = process_batch(face_bbox, vert_geom, edge_geom, face_geom,
                                      edgeFace_adj, edgeVert_adj, faceEdge_adj,
                                      args, class_label, batch_point_data, file_name)
        print(f"Successfully processed {success_count} items out of {b}")

    print('write stl...')
    mesh_tool = Brep2Mesh(input_path=args.save_folder)
    mesh_tool.generate()


if __name__ == '__main__':
    mp.set_start_method('spawn')

    name = 'deepcad'
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file).get(name, {})

    # abc
    # config['faceGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_faceGeom/epoch_3000.pt')
    # config['edgeGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_edgeGeom/epoch_3000.pt')
    # config['vertGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_vertGeom/epoch_2000.pt')
    # config['faceBbox_path'] = os.path.join('upload_checkpoints', name, 'geom_faceBbox/epoch_3000.pt')
    # config['faceEdge_path'] = os.path.join('upload_checkpoints', name, 'topo_faceEdge/epoch_1000.pt')
    # config['edgeVert_path'] = os.path.join('upload_checkpoints', name, 'topo_edgeVert/epoch_500.pt')

    # furniture
    # config['faceGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_faceGeom/epoch_3000.pt')
    # config['edgeGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_edgeGeom/epoch_3000.pt')
    # config['vertGeom_path'] = os.path.join('upload_checkpoints', name, 'geom_vertGeom/epoch_3000.pt')
    # config['faceBbox_path'] = os.path.join('upload_checkpoints', name, 'geom_faceBbox/epoch_3000.pt')
    # config['faceEdge_path'] = os.path.join('checkpoints', name, 'myfulltopo_faceEdge/epoch_2000.pt')
    # config['edgeVert_path'] = os.path.join('upload_checkpoints', name, 'topo_edgeVert/epoch_3000.pt')

    # deepcad
    config['edgeGeom_path'] = os.path.join('upload_checkpoints', name + "_with_pc", 'geom_edgeGeom/epoch_3000.pt')
    config['vertGeom_path'] = os.path.join('upload_checkpoints', name + "_with_pc", 'geom_vertGeom/epoch_3000.pt')
    config['faceBbox_path'] = os.path.join('upload_checkpoints', name + "_with_pc", 'geom_faceBbox/epoch_3000.pt')
    config['faceEdge_path'] = os.path.join('upload_checkpoints', name + "_with_pc", 'topo_faceEdge/epoch_1350.pt')
    config['edgeVert_path'] = os.path.join('upload_checkpoints', name + "_with_pc", 'topo_edgeVert/epoch_1550.pt')
    config['faceGeom_path'] = os.path.join('checkpoints', name, 'geom_faceGeom/epoch_3000.pt')

    config['test_path'] = os.path.join('inference', name + '_complex_test.pkl')
    config['save_folder'] = os.path.join('samples', name + "facetest")
    config['name'] = name
    config['parallel'] = True

    main(args=Namespace(**config))
