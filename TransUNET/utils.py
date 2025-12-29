import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk
from scipy.ndimage import zoom, distance_transform_edt

# ================== 1. 损失函数 ==================
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

# ================== 2. 手动实现 HD95 (替代 medpy) ==================
def compute_hd95(pred, gt, spacing=None):
    """
    使用 scipy 计算 Hausdorff Distance 95%
    pred, gt: bool numpy array
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)

    if pred.sum() == 0 or gt.sum() == 0:
        # 如果预测为空或GT为空，距离定义为0或最大值，这里为了不报错返回0
        return 0.0

    # 计算距离变换图
    # distance_transform_edt 计算非零点到最近零点的欧氏距离
    # 所以我们要对 "非前景" (即背景) 做距离变换
    dt_gt = distance_transform_edt(~gt, sampling=spacing)
    dt_pred = distance_transform_edt(~pred, sampling=spacing)

    # 获取预测边界在GT距离图上的值
    sds_pred = dt_gt[pred]
    # 获取GT边界在预测距离图上的值
    sds_gt = dt_pred[gt]

    # 合并两个方向的距离
    # HD95 是双向 Hausdorff 距离的 95% 分位数
    ns = sds_pred.size + sds_gt.size
    if ns == 0:
        return 0.0
    
    # 简单的单向或双向组合
    hd95 = np.percentile(np.concatenate([sds_pred, sds_gt]), 95)
    return hd95

# ================== 3. 核心评估函数 ==================
def calculate_metrics(pred, target, num_classes=2):
    """
    计算 Batch 级别的评估指标
    Returns: acc, miou, dice, hd95
    """
    pred = pred.data.cpu().numpy()
    target = target.data.cpu().numpy()
    
    batch_acc = []
    batch_miou = []
    batch_dice = []
    batch_hd95 = []

    for i in range(pred.shape[0]):
        p = pred[i]
        t = target[i]
        
        # --- Pixel Accuracy ---
        acc = np.sum(p == t) / (p.size + 1e-8)
        batch_acc.append(acc)
        
        # --- IoU & Dice ---
        class_ious = []
        class_dices = []
        
        for c in range(num_classes):
            p_c = (p == c)
            t_c = (t == c)
            intersection = np.logical_and(p_c, t_c).sum()
            union = np.logical_or(p_c, t_c).sum()
            
            # IoU
            if union == 0:
                iou = 1.0 
            else:
                iou = intersection / (union + 1e-8)
            class_ious.append(iou)
            
            # Dice
            if p_c.sum() + t_c.sum() == 0:
                dice = 1.0
            else:
                dice = 2 * intersection / (p_c.sum() + t_c.sum() + 1e-8)
            class_dices.append(dice)
            
        batch_miou.append(np.mean(class_ious))
        batch_dice.append(np.mean(class_dices))
        
        # --- HD95 (只针对血管类，即 label=1) ---
        try:
            # 传入 bool 类型的 mask
            hd95_val = compute_hd95(p == 1, t == 1)
            batch_hd95.append(hd95_val)
        except Exception:
            batch_hd95.append(0.0)
            
    return np.mean(batch_acc), np.mean(batch_miou), np.mean(batch_dice), np.mean(batch_hd95)

# ================== 4. 测试辅助函数 ==================
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        # 使用 numpy/scipy 实现的 dice
        intersection = (pred * gt).sum()
        dice = (2. * intersection) / (pred.sum() + gt.sum())
        # 使用自定义 hd95
        hd95 = compute_hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list