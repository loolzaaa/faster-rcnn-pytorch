import ast
import torch
import cv2 as cv
import numpy as np


def parse_additional_params(params):
    assert params is not None, 'Additional parameter list cannot be None'
    add_params = {}
    err_params = []
    exclude_list = ['true', 'false']
    if len(params) == 0:
        return add_params, err_params
    else:
        for param in params:
            pair = param.split('=')
            if len(pair) != 2:
                err_params.append(param)
            else:
                key = pair[0]
                if len(key) == 0 or key.lower() in exclude_list:
                    err_params.append(param)
                    continue

                value = pair[1]
                if value.find('[') == 0:
                    try:
                        add_params[key] = list(ast.literal_eval(value))
                    except:
                        err_params.append(param)
                    continue

                if value.find('{') == 0:
                    try:
                        add_params[key] = dict(ast.literal_eval(value))
                    except:
                        err_params.append(param)
                    continue

                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                add_params[key] = value

    return add_params, err_params


def smooth_l1_loss(input, target, beta=1, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2
    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv.putText(im, '%s: %.3f' % (class_name, score),
                       (bbox[0], bbox[1] + 15),
                       cv.FONT_HERSHEY_PLAIN,
                       1.0,
                       (0, 0, 255),
                       thickness=1)
    return im
