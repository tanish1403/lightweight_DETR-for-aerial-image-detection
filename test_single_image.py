import torch
from PIL import Image
import torchvision.transforms as T
from models import build
from datasets.coco import CocoDetection
from main import get_args_parser
import argparse
import cv2
import numpy as np

def visualize_obb(image, boxes, scores, threshold=0.5):
    image_np = np.array(image)
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x, y, w, h, theta = box
            rect = ((x, y), (w, h), theta * 180 / np.pi)
            poly = cv2.boxPoints(rect).astype(np.int32)
            cv2.polylines(image_np, [poly], True, (0, 255, 0), 2)
            cv2.putText(image_np, f"{score:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return Image.fromarray(image_np)

def main(args):
    model, criterion, postprocessors = build(args)
    model.eval()
    device = torch.device(args.device)
    model.to(device)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    dataset = CocoDetection(args.data_path + 'test_dataset/images', args.data_path + 'test_dataset/annotations/annotations.json', T.ToTensor())
    img, target = dataset[0]

    img = img.to(device)
    with torch.no_grad():
        outputs = model([img])

    scores = outputs['pred_logits'].softmax(-1)[0, :, :-1].max(-1)[0]
    boxes = outputs['pred_boxes'][0]  # [num_queries, 5]
    keep = scores > 0.5
    boxes = boxes[keep]
    scores = scores[keep]

    result = visualize_obb(Image.open(args.data_path + 'test_dataset/images/0001.png'), boxes.cpu().numpy(), scores.cpu().numpy())
    result.save('output.jpg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Sparse DETR test script', parents=[get_args_parser()])
    parser.add_argument('--data_path', default='', help='Path to dataset')
    parser.add_argument('--device', default='cpu', help='Device to use')
    parser.add_argument('--resume', default='', help='Path to checkpoint')
    parser.add_argument('--obb', action='store_true', help='Enable OBB mode')
    args = parser.parse_args()
    main(args)