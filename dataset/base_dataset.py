import torch.utils.data as data
import cv2
import numpy as np
import torch
import math
from gaussian_tools import gaussian_radius, draw_umich_gaussian
from transform import ex_box_jaccard

class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h = None, input_w = None, down_ratio = None, transform = False):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.img_ids = None
        self.num_classes = None
        self.max_objs = 500
        self.image_distort = None
        self.transform = transform


    def  load_img_ids(self):
        return None
    def load_imgs(self, index):
        return None
    
    def load_ann_folders(self, img_id):
        return None
    
    def load_annotations(self, index):
        return None

    def dec_evaluation(self, result_path):
        return None
    def data_transform(self, img, anns):
        return None
    
    def __len__(self):
        return len(self.img_ids)
    
    def processing_test(self, image, input_h, input_w):
        image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        out_image = image.astype(np.float32) / 255.
        out_image = out_image-0.5
        out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
        out_image = torch.from_numpy(out_image)
        return out_image
    
    def cal_bbox_wh(self, pts_4):
        x1= np.min(pts_4[:,0])
        y1= np.min(pts_4[:,1])
        x2= np.max(pts_4[:,0])
        y2= np.max(pts_4[:,1])
        return x2-x1, y2-y1
    
    def cal_bbox_pts(self, pts_4):
        
        x1= np.min(pts_4[:,0])
        y1= np.min(pts_4[:,1])
        x2= np.max(pts_4[:,0])
        y2= np.max(pts_4[:,1])
        bl = [x1, y2]
        tl = [x1, y1]
        tr = [x2, y1]
        br = [x2, y2]
        return np.asarray([bl, tl, tr, br], np.float32)
    

    def reorder_pts(self, tt, rr, bb, ll):
        pts = np.asarray([tt, rr, bb, ll], np.float32)
        l_ind = np.argmin(pts[:,0])
        r_ind = np.argmax(pts[:,0])
        t_ind = np.argmin(pts[:,1])
        b_ind = np.argmax(pts[:,1])
        tt_new = pts[t_ind, :]
        rr_new = pts[r_ind, :]
        bb_new = pts[b_ind, :]
        ll_new = pts[l_ind, :]
        return tt_new, rr_new, bb_new, ll_new
    

    def generate_grnd_trth(self, img, ann):
        img = np.asanyarray(np.clip(img, 0, 255), dtype=np.float32)
        if self.image_distort:
            img = self.image_distort(img)
            img = np.asanyarray(np.clip(img, 0, 255), dtype=np.float32)
        img = np.transpose((img/255. -0.5), (2, 0, 1))  
        img_h = self.input_h//self.down_ratio
        img_w = self.input_w//self.down_ratio

        hm = np.zeros((self.num_classes, img_h, img_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 10),dtype=np.float32)

        cls_theta = np.zeros((self.max_objs, 10), dtype=np.float32)          
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)   
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        num_objs = min(ann["rects"].shape[0], self.max_objs)

        for k in range(num_objs):
            rect = ann["rects"][k, :]
            cen_x, cen_y, bbox_w, bbox_h, theta = rect

            radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
            ct = np.array([cen_x, cen_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[ann["cat"][k]], ct_int, radius)
            ind[k] = ct_int[1] * img_w + ct_int[0]
            reg[k] = ct - ct_int
            reg_mask[k] = 1

            pts_4 = cv2.boxPoints(((cen_x, cen_y), (bbox_w, bbox_h), theta))

            bl= pts_4[0,:]
            tl = pts_4[1,:]
            tr = pts_4[2,:]
            br = pts_4[3,:]

            tt =( np.asarray(tl, np.float32)+np.asarray(tr, np.float32))/2
            rr = (np.asarray(tr, np.float32)+np.asarray(br, np.float32))/2
            bb = (np.asarray(br, np.float32)+np.asarray(bl, np.float32))/2
            ll = (np.asarray(bl, np.float32)+np.asarray(tl, np.float32))/2

            if theta in [-90.0, -0.0,0.0]:
                tt, rr, bb, ll = self.reorder_pts(tt, rr, bb, ll)
            
            wh[k, 0:2] = tt-ct
            wh[k, 2:4] = rr-ct
            wh[k, 4:6] = bb-ct
            wh[k, 6:8] = ll-ct

            w_hbbox, h_hbbox = self.cal_bbox_wh(pts_4)
            wh[k, 8:10] = 1.*w_hbbox, 1.*h_hbbox

            jaccard_score = ex_box_jaccard(pts_4.copy(), self.cal_bbox_pts(pts_4).copy())
            if jaccard_score < 0.95:
                cls_theta[k, 0]=1

        ret = {
            'input': img,
            'hm': hm,
            "reg_mask": reg_mask,
            "ind": ind,
            "wh": wh,
            "reg": reg,
            "cls_theta": cls_theta
        }

        return ret
    

    def __getitem__(self, index):
        image = self.load_imgs(index)
        img_h, img_w, _ = image.shape

        if self.phase=="test":
            img_id = self.img_ids[index]
            image = self.processing_test(image, self.input_h, self.input_w)
            return {
                'image': image,
                'img_id': img_id,
                'image_w': img_w,
                'image_h': img_h
            }
        elif self.phase=="train":
            ann = self.load_anns(index)
            if self.transform:
                image, ann = self.data_transform(image, ann)
            data_dict = self.generate_grnd_trth(image, ann)
            return data_dict



            
        