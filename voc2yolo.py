import random
import os
import xml.etree.ElementTree as ET


ann_mode = 0

train_percent = 0.8
val_percent = 0.1
test_percent = 0.1

voc_ann_path = "..\dataset\SCCOS"
voc_ann_sets = ['train', 'val', 'test']

def convert_ann(img_id, list_file, img_set):
    in_file = open(os.path.join(voc_ann_path,  '%s.xml' % (img_id)), encoding='utf-8')
    # print(os.path.exists(os.path.join(voc_ann_path,  '%s.xml' % (img_id))))
    tree=ET.parse(in_file)
    root = tree.getroot()
    pass
    

def train_val_test_ids(train_percent, val_percent, test_percent):
    train_txt = "train_id.txt"
    val_txt = "val_id.txt"
    test_txt = "test_id.txt"
    # creating train, val, test id text files
    
    img_ids = [img_id.split('.')[0] for img_id in os.listdir(os.path.join(voc_ann_path)) if img_id.endswith('.xml')]
   
    random.shuffle(img_ids)
    
    total = len(img_ids)
    trainlim = int(total * train_percent)
    vallim = int(total * val_percent) + trainlim

    train_ids = img_ids[:trainlim]
    val_ids = img_ids[trainlim:vallim]
    test_ids = img_ids[vallim:]

    with open(train_txt,"w") as f:
        for imgs in train_ids:
            f.write(imgs + "\n")

    with open(val_txt,"w") as f:        
        for imgs in val_ids:
            f.write(imgs + "\n")

    with open(test_txt,"w") as f:     
        for imgs in test_ids:
            f.write(imgs + "\n")
    return

if __name__ == "__main__":
    random.seed(0)
    
    if os.path.exists("train_id.txt") and os.path.exists("val_id.txt") and os.path.exists("test_id.txt"):
        print("train, val, test id files already exists")
    else:
        train_val_test_ids(train_percent, val_percent, test_percent)
        print("train, val, test id files created")

    for img_set in voc_ann_sets:
        
        img_ids = open(img_set + "_id.txt", encoding='utf-8').read().strip().split()
        list_file = open(img_set + ".txt", 'w', encoding='utf-8') 
        
        for img_id in img_ids:
            list_file.write(os.path.join(voc_ann_path, '%s.jpg\n' % (img_id)))

            convert_ann(img_id, list_file, img_set)
            list_file.write('\n')
        list_file.close()
        # delete id file
        
        os.remove(img_set + "_id.txt")
        print(img_set + " list file created")



