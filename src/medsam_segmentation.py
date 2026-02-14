from typing import Dict, Tuple, List
import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.metrics import compute_surface_dice
from skimage import transform
import torch.nn.functional as F
import os
import pickle


try:
    from data_io import ImageData
except ImportError:
    from src.data_io import ImageData

from segment_anything import build_sam_vit_b

def medsam_inference(medsam_model, img_embed, box_torch, H, W, batch_size):
    print("\nInside medsam_inference()")
    all_predictions = []

    for i in range(0, img_embed.size(0), batch_size):
        print(f"Processing batch {i // batch_size + 1}...")

        img_embed_batch = img_embed[i:i + batch_size]
        box_torch_batch = box_torch[i:i + batch_size]

        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points=None,
            boxes=box_torch_batch,
            masks=None,
        )

        low_res_logits, _ = medsam_model.mask_decoder(
            image_embeddings=img_embed_batch,
            image_pe=medsam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        low_res_pred = torch.sigmoid(low_res_logits)  # (B, 1, 256, 256)

        for j in range(low_res_pred.shape[0]):
            pred = low_res_pred[j].squeeze().detach().cpu().numpy()  # (H, W)
            all_predictions.append(torch.from_numpy(pred).unsqueeze(0)) # (1, H, W)

    return torch.stack(all_predictions, dim=0) # (N, 1, H, W)

class MedSAMTool():
    """
    MedSAMTool is a class that provides a simple interface for the MedSAM model
    """

    def __init__(self, checkpoint_path, model_name='medsam', gpu_id: int = 0, **kwargs):
        self.checkpoint_path = checkpoint_path
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            print(f"Warning: GPU ID {gpu_id} is not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            
        self.kwargs = kwargs
    
    def predict(self, images: ImageData, scaled_boxes, used_for_baseline, is_rgb=True) -> np.ndarray:
        medsam_model = build_sam_vit_b(checkpoint=self.checkpoint_path)
        medsam_model.to(self.device)
        medsam_model.eval()
       
        resized_imgs = images.raw
        if used_for_baseline and is_rgb:
            print("Is RGB - normalizing but not clipping.")
            # "if they were already within the expected intensity range of [0, 255], their intensities remained unchanged. However, if they fell outside this range, we utilized max-min normalization to rescale the intensity values to [0, 255]"
            # based from: https://github.com/bowang-lab/MedSAM/blob/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/utils/pre_grey_rgb.py#L49
            for i in range(len(resized_imgs)):
                img_np = resized_imgs[i]
                val_range = np.max(img_np) - np.min(img_np)
                if val_range == 0:
                    resized_imgs[i] = np.zeros_like(img_np, dtype=np.uint8)
                else:
                    resized_imgs[i] = np.uint8((img_np - img_np.min()) / val_range * 255.0)
        if used_for_baseline and not is_rgb:
            print("Is not RGB - normalizing AND clipping.")
            for i in range(len(resized_imgs)):
                img_np = resized_imgs[i]
                # "we clipped the intensity values to the range between the 0.5th and 99.5th percentiles before rescaling them to the range of [0, 255]"
                # based from: https://github.com/bowang-lab/MedSAM/blob/d71e8a1a99ad751840a22a7fa3ecfb4166fb1488/utils/pre_grey_rgb.py#L59C9-L63C50
                positive_pixels = img_np[img_np > 0]
                if positive_pixels.size == 0:
                    resized_imgs[i] = np.zeros_like(img_np, dtype=np.uint8)
                    continue
                lower_bound, upper_bound = np.percentile(positive_pixels, 0.5), np.percentile(positive_pixels, 99.5)
                img_np_pre = np.clip(img_np, lower_bound, upper_bound)
                val_range = np.max(img_np_pre) - np.min(img_np_pre)
                if val_range == 0:
                    img_np_pre = np.zeros_like(img_np_pre)
                else:
                    img_np_pre = (img_np_pre - np.min(img_np_pre)) / val_range * 255.0
                img_np_pre[img_np == 0] = 0
                resized_imgs[i] = np.uint8(img_np_pre)

        img_batch = torch.stack([
            torch.tensor(img).float().permute(2, 0, 1)  # (3, H, W)
            for img in resized_imgs
        ]).to(self.device)  # (B, 3, H, W)
        
        batch_size = images.batch_size
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, img_batch.size(0), batch_size):
                print(f"Processing batch {i // batch_size + 1}...")
                batch = img_batch[i:i+batch_size]
                temp_image_embedding = medsam_model.image_encoder(batch)  # (B, 256, 64, 64)
                all_embeddings.append(temp_image_embedding)

        image_embeddings = torch.cat(all_embeddings, dim=0)  # (B, 256, 64, 64)

        scaled_boxes = np.array(scaled_boxes)
        scaled_boxes = torch.tensor(scaled_boxes).float()
        scaled_boxes = scaled_boxes.to(self.device)

        torch.cuda.empty_cache()
        medsam_seg = medsam_inference(medsam_model, image_embeddings, scaled_boxes, 512, 512, images.batch_size)
        return medsam_seg
    
    def evaluate(self, pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
        resized_preds = []
        for i in range(len(pred_masks)):
            H, W = gt_masks[i].shape
            resized_pred = transform.resize(
                pred_masks[i].cpu().numpy(), (H, W), order=3, preserve_range=True, anti_aliasing=True
            ).astype(np.uint8)
            resized_preds.append(resized_pred)
            
        print("\nEvaluating predictions...")
        spacing= (1.0, 1.0)  # Assuming isotropic spacing for simplicity
        tolerance = 2.0
        
        total_dsc, total_nsd = 0, 0
        for i, (pred, gt) in enumerate(zip(resized_preds, gt_masks)):
            gt_tensor = torch.tensor(gt).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
            pred_tensor = torch.tensor(pred).unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)

            dice_metric = DiceMetric(include_background=False, reduction="mean")
            immediate_dsc_metric = dice_metric(pred_tensor, gt_tensor)

            immediate_nsd_metric = compute_surface_dice(pred_tensor, gt_tensor, class_thresholds=[tolerance], spacing=spacing)
            print(f"{i} | DSC: {round(immediate_dsc_metric.item(), 6)} | NSD: {round(immediate_nsd_metric.item(), 6)}")
            
            total_dsc += immediate_dsc_metric
            total_nsd += immediate_nsd_metric

        print("\n=======================")
        print(f"Average DSC metric: {round(total_dsc.item() / len(pred_masks), 6)}")
        print(f"Average NSD metric: {round(total_nsd.item() / len(pred_masks), 6)}")
        return {"dsc_metric": total_dsc.item() / len(pred_masks),
                "nsd_metric": total_nsd.item() / len(pred_masks)}
    
    def preprocess(self, image_data: ImageData) -> ImageData:
        return image_data
    
    def loadData(self, data_path: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        with open(data_path, "rb") as f:
            imgs, boxes, masks = pickle.load(f)
        return imgs, boxes, masks