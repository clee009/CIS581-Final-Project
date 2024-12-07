from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models, load_hmr2, DEFAULT_CHECKPOINT
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
import pdb
import json

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

def human_pose_estimation(img_folder, render_img, side_view, top_view):
    import time
    start = time.time()
    # parser = argparse.ArgumentParser(description='HMR2 demo code')
    # parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    # parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    # parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    # parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    # parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    # parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    # parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    # parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    # parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    # parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    # args = parser.parse_args()

    # Download and load checkpoints
    download_models(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    from detectron2.config import LazyConfig
    import hmr2
    cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
   
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Get all demo images that end with .jpg or .png
    # img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])
    file_type = ['*.jpg', '*.png']
    img_paths = sorted([img for end in file_type for img in Path(img_folder).glob(end)],key=lambda x: int(''.join(filter(str.isdigit, x.stem))))  # Extract numeric part of filename

    # save json for facing orientations
    facing_direction_whole_dataset = {}

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5) 
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        
        for batch in dataloader:
            batch = recursive_to(batch, device)
            t1 = time.time()
            with torch.no_grad():
                out = model(batch)
            t2 = time.time()
            print(t2-t1)
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                if render_img: # Render the result
                    # Get filename from path img_path
 
                    white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                    input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                    input_patch = input_patch.permute(1,2,0).numpy()

                    regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            root_orientation = out['pred_smpl_params']['global_orient'],
                                            )

                    final_img = np.concatenate([input_patch, regression_img], axis=1)

                    if side_view:
                        side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True,
                                                root_orientation = out['pred_smpl_params']['global_orient'])
                        final_img = np.concatenate([final_img, side_img], axis=1)

                    if top_view:
                        top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                                out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                top_view=True,
                                                root_orientation = out['pred_smpl_params']['global_orient'])
                        final_img = np.concatenate([final_img, top_img], axis=1)

                    # cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                    # Add all verts and cams to list
                    verts = out['pred_vertices'][n].detach().cpu().numpy()
                    cam_t = pred_cam_t_full[n]
                    all_verts.append(verts)
                    all_cam_t.append(cam_t)

        misc_args = dict(
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        facing_direction_2d, facing_direction_3d, position_2d = renderer.direction_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)
        

        det_jpg_name = str(img_path).split("/")[-1]
        subdict = {}
        subdict["facing_direction_2d"] = facing_direction_2d
        subdict["facing_direction_3d"] = facing_direction_3d
        subdict["position_2d"] = position_2d
        facing_direction_whole_dataset[det_jpg_name] = subdict

        end = time.time()
        print(end - start)






    # # Save to JSON
    # with open("facing_directions.json", "w") as json_file:
    #     json.dump(facing_direction_whole_dataset, json_file, indent=4)
    return facing_direction_whole_dataset

