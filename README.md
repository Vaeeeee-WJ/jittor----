# 第四届计图人工智能挑战赛-赛题一：开放域少样本视觉分类赛题

## 简介

本项目基于Jittor框架，根据[赛题一](https://www.educoder.net/competitions/Jittor-5)中的内容和要求，使用预训练的CLIP模型和极少的多领域训练样本完成的一些工作。

比赛数据集由以下四个子数据集构成（Tsinghua-Dog数据集，Caltech-101数据集，Food-101数据集，动物分类自建数据集），共374个类别。对于每个类别，选手可以从训练集中挑出任意4张图片训练自己的模型，当训练结束后，对测试集的每张图片进行分类，输出每张图片的Top5分类。
赛题baseline基于CLIP，本次比赛提供了两个版本的baseline（Training和Training-Free）,如下图所示：

![clip.png](/image/README/1724316686383.png)
**训练集及Baseline**

- [Baseline](https://github.com/uyzhang/JCLIP)
- [TrainSet.zip](https://cloud.tsinghua.edu.cn/f/7c44b138a6344f4b8fd1/?dl=1)
- [classname.txt](https://cloud.tsinghua.edu.cn/f/418b311c5ae8484f8208/?dl=1)
- [train.txt](https://cloud.tsinghua.edu.cn/f/212edd1e7b3b44f5b301/?dl=1)
- [RN101.pkl](https://pan.baidu.com/s/1bvTAav9-TvXjn-lcX_yVDw?pwd=dv7r)

> （**备注**：Baseline中并没有给 `RN101.pkl`文件，只给了转换脚本可以将 `PyTorch`权重转换为 `Jittor`权重，但是该脚本必须要求环境中同时安装 `PyTorch`和 `Jittor`的cuda版本，否则运行会报错。如果你是在不同环境中安装的这两个框架，可以参考[`self.ipynb`](/JCLIP/self.ipynb)进行权重的转换，或者点击上面链接直接下载转换好的权重文件。）

## 安装

本项目可在一张NVIDIA GeForce RTX 3090上运行

#### 运行环境

- Windows 10
- python == 3.9.16
- jittor == 1.3.8.5

#### 安装依赖

执行以下命令安装 jittor

```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple jittor
```

安装其他依赖库：

```
pip install -r requirements.txt
```

## 数据预处理

- 从TrainSet数据集当中随机挑选出了3000张图片作为个人的验证集（命名为：TestSetZ数据集），用于提交结果之前简单评估模型的微调效果。可以点击这里直接下载[TestSetZ](https://pan.baidu.com/s/1u2wJIxbnfmnPaI9-1-9TEg?pwd=pg9b)，或者在 [`self.ipynb`](/JCLIP/self.ipynb)文件中运行对应的代码即可生成。
- 为了提高模型的泛化能力，对训练集(374*4张)采用了**随机裁剪**和**随机擦除**的数据增强方法，其他一些方法尝试后发现反而会降低模型的效果。

## 思路

- 首先使用 `ViT-B-32.pkl`和 `RN101.pkl`两个预训练模型在训练集上进行训练和简单的微调，然后选择出效果更好的模型；然后在该模型基础上使用不同的微调方法，如 `Linear Probe`，`Tip-Adapter`等，进行二次训练和微调，最终对不同方法得到的模型集成融合。
  ![idea.png](/image/README/idea.png)
- 尽管选择多个模型融合之后可以提高模型在测试集上的准确率，但是最终效果受各个模型的权重系数影响很大，并且发现Tip-Adapter方法得到clip模型在训练集以外的样本上表现的较差，
  集成时需要设置一个非常小的权重系数，但是对于已知类别，模型的泛化能力表现的较好。所以最终选择 **经过训练的CLIP模型 + 使用Tip-Adapter方法微调的CLIP模型** 作为最终的模型。

### 微调方法

#### 1️⃣ 微调Prompt

- **prompt中考虑在类别名称后面附加文本长度信息**，在测试集上的准确率得到一定提升。即将prompt从 `A photo of XXX` 替换成成 `A photo of XXX with xxx_length`，其中 `XXX`表示类别名称，`xxx_length`表示类别名称的文本长度。例如：`A photo of ant with 3` 、  `A photo of Bernese_mountain_dog with 20`等。
  最终选择 `A photo of XXX and the length of the prompt is xxx_length`作为所有类别的文本描述。更多微调的一些方式可参考：[`generate_prompt`](./JCLIP/utils.py#L25)

以下是部分微调的一些结果：

> - 2024/5/10：训练了50个epoch，最终在TestSetZ的准确率为：0.6923，TestSetA上的准确率为0.6541；
> - 2024/5/11：微调学习率，训练200个epoch，最终在TestSetZ的准确率为：0.7220，TestSetA上的准确率为0.6772；
> - 2024/5/11：微调学习率，训练250个epoch，最终在TestSetZ的准确率为：0.7243，TestSetA上的准确率为0.6769；

#### 2️⃣Linear Probe

- 冻住主干，仅微调一个线性分类器，经过测试发现，该方法效果并不好。

#### 3️⃣Tip-Adapter

- 主要利用了测试图像和训练集图像之间的相似度关系。
  ![cache](/image/README/cache_model.png)

#### 4️⃣Cross-modal-Adaptation

- 将多种模态的信息融合在一起，即将每张图像的图像特征和文本特征视作同一个特征来进行训练。

#### 5️⃣FD-Align

- 一种不影响模型对虚假特征识别能力的微调方法。

#### 6️⃣WiSE-FT

- 通过线性插值的方式，将微调后的模型与原始零样本模型进行权重空间集成。

## 训练

- 下载对应的数据集和文件，项目文件夹结构如下：

```
├── root
│   ├── JCLIP
│   |   ├── configs
│   |       ├── no_finetune.yaml
│   |       ├── no_finetune_v1.yaml
│   |       ├── Tip-Adapter-F.yaml
│   |       ├── Coop.yaml
│   |       ├── cross-modal-Adapter.yaml
│   |       ├── FD-Align.yaml
│   |   ├── finetune
│   |       ├── cache_module.py
│   |       ├── Coop.py
│   |       ├── Cross_modal_Adapter.py
│   |       ├── FD_Align.py
│   |       ├── Tipadapter.py
│   |   ├── jclip
│   |   ├── train.py
│   |   ├── test.py
│   |   ├── utils.py
│   |   ├── train1.sh
│   |   ├── train2.sh
│   |   ├── self.ipynb
│   ├── TestSetA
│   ├── TestSetB
│   ├── TestSetZ
│   ├── Weights
│   |   ├── initial_weight
│   |       ├── RN101.pkl
│   |       ├── ViT-B-32.pkl
│   ├── results
│   ├── train.txt
│   ├── train_4class.txt
│   ├── TestSetZ-label.txt
```

- 执行以下步骤进行训练：

1. 将 `no_finetune.yaml`  预训练模型的路径设置成 `ViT-B-32.pkl`所在路径，运行命令：``sh train1.sh``
1. 将 `Tip-Adapter-F.yaml` 中预训练模型的路径设置成上一个步骤得到的权重所在路径，运行命令：``sh train2.sh``

## 推理

- 在 `test.py`文件中设置你的根目录(`root_TrainSet`)，`pkl_path`和 `Tip_Adapter`分别为步骤1、2得到的权重路径，运行命令：``python test.py``，最终结果文件保存在`results`文件夹下。
- 如果要使用其他方法更改设置`method_name`的值即可。


## 引用

**Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling**

```bash
@article{zhang2021tip,
  title={Tip-Adapter: Training-free CLIP-Adapter for Better Vision-Language Modeling},
  author={Zhang, Renrui and Fang, Rongyao and Gao, Peng and Zhang, Wei and Li, Kunchang and Dai, Jifeng and Qiao, Yu and Li, Hongsheng},
  journal={arXiv preprint arXiv:2111.03930},
  year={2021}
}
```

**Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models**

```bash
@misc{lin2023crossmodal,
  title={Multimodality Helps Unimodality: Cross-Modal Few-Shot Learning with Multimodal Models},
  author={Lin, Zhiqiu and Yu, Samuel and Kuang, Zhiyi and Pathak, Deepak and Ramanan, Deva},
  year={2023},
  eprint={2301.06267},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

**FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning**

```bash
@article{song2023FD,
    title={FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning},
    author={Kun Song and Huimin Ma and Bochao Zou and Huishuai Zhang and Weiran Huang},
    journal={NeurIPS},
    year={2023}
}
```

**Learning to Prompt for Vision-Language Models**

```bash
@article{zhou2022coop,
    title={Learning to Prompt for Vision-Language Models},
    author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
    journal={International Journal of Computer Vision (IJCV)},
    year={2022}
}
```

**Robust fine-tuning of zero-shot models**

```bash
@article{wortsman2021robust,
  title={Robust fine-tuning of zero-shot models},
  author={Wortsman, Mitchell and Ilharco, Gabriel and Kim, Jong Wook and Li, Mike and Kornblith, Simon and Roelofs, Rebecca and Gontijo-Lopes, Raphael and Hajishirzi, Hannaneh and Farhadi, Ali and Namkoong, Hongseok and Schmidt, Ludwig},
  journal={arXiv preprint arXiv:2109.01903},
  note={\url{https://arxiv.org/abs/2109.01903}},
  year={2021}
}
```
