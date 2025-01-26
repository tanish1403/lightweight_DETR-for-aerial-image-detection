import os
import xml.etree.ElementTree as ET

def extract_classes(ann_path,classes_file): 
    classes = set()

    for filename in os.listdir(ann_path):
        if filename.endswith('.xml'):
            xml_file = open(os.path.join(ann_path, filename), encoding='utf-8')
            tree=ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.iter('object'):
                cls = obj.find('name').text
                classes.add(cls)

    with open(classes_file, 'w') as f:
        for cls in classes:
            f.write(cls + '\n') 
        f.close()
    return classes



def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)