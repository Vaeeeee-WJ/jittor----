import random
import yaml
import jittor as jt
import numpy as np
import pandas as pd
from PIL import Image
import jclip as clip
from jittor.transform import ImageNormalize, Compose
from datetime import datetime
from jittor.optim import Optimizer

# 类别到idx的映射
def class_2_idx(path):
    '''
    path   : classes.txt文件路径
    return : dict: {'Bear': 0,'Bee': 1, ..., 'papillon': 373,...}
    '''
    df_classes   = pd.read_csv(path, delimiter = ' ', header = None, index_col = False)
    list_classes = list(map(lambda x: x.split('_',1)[-1], df_classes[0]))
    res = {k:v for k,v in zip(list_classes,range(len(list_classes)))}
    
    return res


def generate_prompt(name :str):
    '''
    用于生成提示语句
    '''
    li = [name,
        f"{name} , {len(name)}",
        f"A photo of a {name}",
        f"A photo of a {name} with {len(name)}",
        f"A photo of {name} and the length of the prompt is {len(name)}",
        f"A photo of {name} and the length of the {name} is {len(name)}",
        f"A photo of {name.replace('_', ' ')} and the length of the name is {len(name.replace('_', ' '))}",
        f"A photo of {name} with {len(name)} and _ in {name}" if '_' in name else f"A photo of {name} with {len(name)} and _ not in {name}",
        f"A photo of {name}, {name}, {name}"]


    return li[4]


def normalize_tensor(tensor, a=0.0, b=1.0):
    '''
    归一化
    '''
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    normalized_tensor = a + (tensor - tensor_min) * (b - a) / (tensor_max - tensor_min)
    
    return normalized_tensor


def get_val_text_features(classes_path, model):
    '''
    用于获取模型在待预测类别上的文本特征
    '''
    df_classes     = pd.read_csv(classes_path , delimiter = ' ' , header = None , index_col = False)
    classes        = list(map(lambda x: generate_prompt(x.split('_',1)[-1]), df_classes[0]))
    text           = clip.tokenize(classes)
    with jt.no_grad():
        text_features  = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)     # shape -> (374, 512)  包含了所有类别的文本特征
    return text_features                             


# 自定义的随机擦除函数：类似PyTorch中的torchvision.transforms.RandomErasing函数
class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img

        img = np.array(img)
        h, w, c = img.shape
        area = h * w

        for attempt in range(100):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            erase_h = int(round(np.sqrt(target_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(target_area / aspect_ratio)))

            if erase_h < h and erase_w < w:
                x1 = random.randint(0, h - erase_h)
                y1 = random.randint(0, w - erase_w)

                if self.value == 'random':
                    img[x1:x1+erase_h, y1:y1+erase_w, :] = np.random.randint(0, 256, (erase_h, erase_w, c), dtype=np.uint8)
                else:
                    img[x1:x1+erase_h, y1:y1+erase_w, :] = self.value

                return jt.array(img)

        return jt.array(img)





# 计算模型的参数量 : 根据赛题要求不能超过 500Mb 
def count_parameters_in_mb(model):
    total_params = 0
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count

    total_params_in_mb = total_params / 1e6  # 转换为百万参数（Mb）
    
    assert total_params_in_mb <= 500, "Model {} has too many parameters: {:.2f} Mb. The limit is 500 Mb.".format(model.__class__.__name__, total_params_in_mb)
    # print("Model {} : {:.2f} Mb".format(model.__class__.__name__, total_params_in_mb))
    
    return total_params_in_mb



# 格式化输出args对象中除了 exclude_keys 以外的键值对
def format_args(args, exclude_keys=['root_TrainSet', 'train_path', 'save_path', 'TestSetZ_path', 'label_path', 'classes_path', 'classes_b_path', 'class_4_path']):
    
    max_key_length = max(len(key) for key in vars(args).keys())     # 计算所有键的最大长度
    formatted_args = "\n".join([f"{key.ljust(max_key_length)} : {value}" for key, value in vars(args).items() if key not in exclude_keys]) # 格式化键值对
    
    return formatted_args





# 重写clip.py文件中【_transform方法】，,主要更新了对图像的裁剪的过程
class Image_Transform():
    def __init__(self, img:Image, clip_init):
        self.img  = clip.clip.Resize(224, mode=Image.BICUBIC)(img)  
        self.size = self.img.size
        self.clip = clip_init 
        
    def choose_best_img(self):
    
        # 若图像的长宽比在指定的阈值以内,则直接缩放;否则就裁剪多张图片送入CLIP中，选取预测类别的众数中预测概率值最大的那张
        # 比如同一张图片裁剪了10张图片，预测到了8个A，1个B和1个C，那么选择8个A中概率值最大的那张
        if self.is_direct_scaling(self.size):
            img = self.img.resize((224,224))
            img = self.transform()(img)
            return img
        
        else:
            self.class_path= r'F:\jittor_comprtition\Competition1\classes.txt'
            classes        = list(map(lambda x:generate_prompt(x.split('_',1)[-1]),list(class_2_idx(self.class_path).keys())))
            text           = clip.tokenize(classes)
            with jt.no_grad():
                text_features  = self.clip.encode_text(text)
                text_features /= text_features.norm(dim=1, keepdim=True)
                self.text_features = text_features

            crop_list = self.crop_image_sliding(self.img, 10)

            img_tensor = []
            for img in crop_list:
                img_tensor.append(self.transform()(img))
            img_tensor = jt.array(img_tensor)    # [crop_num, 3, 224 224]
            with jt.no_grad():
                image_features    = self.clip.encode_image(img_tensor)
                image_features   /= image_features.norm(dim=1, keepdim=True)
                text_probs = (100.0 * image_features @ self.text_features.transpose(0, 1)).softmax(dim=-1)
                pred_probs, top_labels = text_probs.topk(1)

                # 选取预测类别的众数且预测概率值最大的那张图片,如果有多个众数，则随机选择一个
                idx, _, idx_counts = jt.unique(top_labels, return_inverse=True, return_counts=True) # 统计众数及其数量
                idx_mode           = idx[jt.argmax(idx_counts, dim=0)[0]]   
                index_mode         = jt.equal(idx_mode, top_labels.flatten()).nonzero().flatten()
                best_img_index     = index_mode[jt.argmax(pred_probs[index_mode],dim=0)[0]].item()
                best_img           = img_tensor[best_img_index]
                return best_img



    def is_direct_scaling(self, img_size:tuple, threshold=1.05):
        aspect_ratio = max(img_size) / 224.0
        return aspect_ratio <= threshold

    def transform(self):
        return Compose([clip.clip._convert_image_to_rgb,
            ImageNormalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
            clip.clip.ImageToTensor()
        ])
    
    def predict(self, img_fea):
        text_probs = (100.0 * img_fea @ self.text_features.transpose(0, 1)).softmax(dim=-1)

    
    def crop_image_sliding(self, img, crop_num, step=None):

        width, height = img.size
        
        # 检查维度，确定沿哪个维度滑动裁剪
        if width > 224:
            slide_dim = 'width'
            fixed_dim = 224
        else:
            slide_dim = 'height'
            fixed_dim = 224

        
        # 计算默认步长，如果没有指定
        if step is None:
            if slide_dim == 'width':
                step = (width - 224) // (crop_num - 1)
            else:
                step = (height - 224) // (crop_num - 1)
        
        cropped_images = []
        
        for i in range(crop_num):
            if slide_dim == 'width':
                left = i * step
                if left + 224 > width:
                    left = width - 224
                box = (left, 0, left + 224, 224)
            else:
                top = i * step
                if top + 224 > height:
                    top = height - 224
                box = (0, top, 224, top + 224)
            
            cropped_img = img.crop(box)
            cropped_images.append(cropped_img)
        
        return cropped_images


# 获取当天日期
def get_date_format():
    today = datetime.now()
    formatted_date =  str(today.month).zfill(2) + str(today.day).zfill(2)
    return formatted_date


# 从yaml文件中读取参数
def load_yaml_params(yaml_path, args):
    with open(yaml_path, 'rb') as f:
        config = yaml.safe_load(f)
    for key in config.keys():
        assert hasattr(args, key), f"The 'args' object does not have the attribute: {key}"
        setattr(args, key, config[key])
    return args

    

