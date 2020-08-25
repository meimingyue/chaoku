#  一、简介

根据上传的图片生成情话，改编自AI Studio项目：https://aistudio.baidu.com/aistudio/projectdetail/738634

小王喜欢旅游，走遍了大江南北，看了不少美景，当然美景旁往往也有很多美女，但是因为嘴很笨一直也不敢搭讪，所以母胎SOLO到现在，我们来帮助一下他吧

小白初入门，谢谢的大家批评指正

本项目通过AI Studio(基于百度深度学习平台飞桨的人工智能学习与实训社区，提供在线编程环境、免费GPU算力、海量开源算法和开放数据，帮助开发者快速创建和部署模型)运行。

本项目AI Studio项目地址：https://aistudio.baidu.com/aistudio/projectdetail/757031

## PaddleHub简介

PaddleHub是飞桨生态的预训练模型应用工具，开发者可以便捷地使用高质量的预训练模型结合Fine-tune API快速完成模型迁移到部署的全流程工作。PaddleHub提供的预训练模型涵盖了图像分类、目标检测、词法分析、语义模型、情感分析、视频分类、图像生成、图像分割、文本审核、关键点检测等主流模型。更多详情可查看PaddleHub官网：https://www.paddlepaddle.org.cn/hub
PaddleHub以预训练模型应用为核心具备以下特点：
1.模型即软件： 通过Python API或命令行实现模型调用，可快速体验或集成飞桨特色预训练模型。
2.易用的迁移学习： 通过Fine-tune API，内置多种优化策略，只需少量代码即可完成预训练模型的Fine-tuning。
3.一键模型转服务： 简单一行命令即可搭建属于自己的深度学习模型API服务完成部署。
4.自动超参优化： 内置AutoDL Finetuner能力，一键启动自动化超参搜索。

# 二、准备工作

1、导入包和两个模型（图像分类和情话生成模型）


```python
!pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple
```


```python
!hub install xception71_imagenet==1.0.0 #图像分类
!hub install ernie_gen_lover_words==1.0.1 #情话生成模型
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    2020-08-25 22:35:23,538-INFO: font search path ['/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/ttf', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/afm', '/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts']
    2020-08-25 22:35:23,864-INFO: generated new fontManager
    Downloading xception71_imagenet
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpmfylo4m3/xception71_imagenet
    [==================================================] 100.00%
    Successfully installed xception71_imagenet-1.0.0
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/sklearn/externals/joblib/externals/cloudpickle/cloudpickle.py:47: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp
    Downloading ernie_gen_lover_words
    [==================================================] 100.00%
    Uncompress /home/aistudio/.paddlehub/tmp/tmpx6j5wgy0/ernie_gen_lover_words
    [==================================================] 100.00%
    Successfully installed ernie_gen_lover_words-1.0.1


**查看一下示例图片**


```python
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 

# 展示美景图片
img1 = mpimg.imread("beauty1.jpg") 

plt.figure(figsize=(10,10))
plt.imshow(img1) 
 
plt.axis('off') 
plt.show()
```


![png](output_9_0.png)


# 三、具体实现

1. 获取图片的分类，即该图片是什么


```python
import paddlehub as hub

module_image = hub.Module(name="xception71_imagenet") # 调用图像分类的模型

test_img_path = "beauty1.jpg" # 选择图像，即要根据哪张图片写诗

# set input dict
input_dict = {"image": [test_img_path]}

# execute predict and print the result
results_image = module_image.classification(data=input_dict)

PictureClassification = list(results_image[0][0].keys())[0]
print("该图片的分类是：",PictureClassification) # 得到分类结果
```

    [32m[2020-08-25 22:50:26,659] [    INFO] - Installing xception71_imagenet module[0m
    [32m[2020-08-25 22:50:26,781] [    INFO] - Module xception71_imagenet already installed in /home/aistudio/.paddlehub/modules/xception71_imagenet[0m
    [32m[2020-08-25 22:50:27,606] [    INFO] - 638 pretrained paramaters loaded by PaddleHub[0m


    该图片的分类是： alp


2. 英文翻译为中文


```python
#导入翻译包#
!pip install translate
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting translate
      Downloading https://mirror.baidu.com/pypi/packages/85/b2/2ea329a07bbc0c7227eef84ca89ffd6895e7ec237d6c0b26574d56103e53/translate-3.5.0-py2.py3-none-any.whl
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from translate) (1.21.0)
    Collecting tox (from translate)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/a4/eb/09b02408f46cd18cc8345849dbf42a7d823ebe1cc8522517dc7b1a924ce5/tox-3.19.0-py2.py3-none-any.whl (83kB)
    [K     |████████████████████████████████| 92kB 16.2MB/s eta 0:00:01
    [?25hRequirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from translate) (2.22.0)
    Collecting lxml (from translate)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/de/3c/fa420469c0d4f62ae39f19ee6505f90d00ae469f6264f4f54e61ed9d9a2c/lxml-4.5.2-cp37-cp37m-manylinux1_x86_64.whl (5.5MB)
    [K     |████████████████████████████████| 5.5MB 26.6MB/s eta 0:00:01
    [?25hRequirement already satisfied: click in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from translate) (7.0)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (1.3.4)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (16.7.9)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (2.0.1)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (1.3.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (0.23)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (0.10.0)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (5.1.2)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (1.4.10)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->translate) (1.15.0)
    Requirement already satisfied: pluggy>=0.12.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tox->translate) (0.13.1)
    Collecting filelock>=3.0.0 (from tox->translate)
      Downloading https://mirror.baidu.com/pypi/packages/93/83/71a2ee6158bb9f39a90c0dea1637f81d5eef866e188e1971a1b1ab01a35a/filelock-3.0.12-py3-none-any.whl
    Collecting py>=1.4.17 (from tox->translate)
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/68/0f/41a43535b52a81e4f29e420a151032d26f08b62206840c48d14b70e53376/py-1.9.0-py2.py3-none-any.whl (99kB)
    [K     |████████████████████████████████| 102kB 34.0MB/s ta 0:00:01
    [?25hCollecting packaging>=14 (from tox->translate)
      Downloading https://mirror.baidu.com/pypi/packages/46/19/c5ab91b1b05cfe63cccd5cfc971db9214c6dd6ced54e33c30d5af1d2bc43/packaging-20.4-py2.py3-none-any.whl
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->translate) (1.25.6)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->translate) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->translate) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->translate) (3.0.4)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->translate) (0.6.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from packaging>=14->tox->translate) (2.4.2)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->translate) (7.2.0)
    Installing collected packages: filelock, py, packaging, tox, lxml, translate
    Successfully installed filelock-3.0.12 lxml-4.5.2 packaging-20.4 py-1.9.0 tox-3.19.0 translate-3.5.0



```python
# from translate import Translator

# translator = Translator(to_lang="chinese")
# PictureClassification_ch = translator.translate("{}".format(PictureClassification))

# print("该图片的分类是：",PictureClassification_ch)
```

    该图片的分类是： alp


alp是阿尔卑斯山，可见翻译不准确，采用百度翻译API，可见https://api.fanyi.baidu.com/doc/11


```python
#百度通用翻译API,不包含词典、tts语音合成等资源，如有相关需求请联系translate_api@baidu.com
# coding=utf-8

import http.client
import hashlib
import urllib
import random
import json

appid = ''  # 填写你的appid
secretKey = ''  # 填写你的密钥

httpClient = None
myurl = '/api/trans/vip/translate'

fromLang = 'auto'   #原文语种
toLang = 'zh'   #译文语种
salt = random.randint(32768, 65536)
q= PictureClassification
sign = appid + q + str(salt) + secretKey
sign = hashlib.md5(sign.encode()).hexdigest()
myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
salt) + '&sign=' + sign

try:
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
    httpClient.request('GET', myurl)

    # response是HTTPResponse对象
    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    result = json.loads(result_all)

    print (result)

except Exception as e:
    print (e)
finally:
    if httpClient:
        httpClient.close()
```

    {'from': 'en', 'to': 'zh', 'trans_result': [{'src': 'alp', 'dst': '阿尔卑斯山'}]}


3、生成情话


```python
PictureClassification_ch = result.get('trans_result')[0].get('dst')
module = hub.Module(name="ernie_gen_lover_words")
test_texts = []
test_texts.append(PictureClassification_ch)
results = module.generate(texts=test_texts, use_gpu=True, beam_width=5)
for result in results:
    print(result)
```

    [32m[2020-08-25 23:36:40,138] [    INFO] - Installing ernie_gen_lover_words module[0m
    [32m[2020-08-25 23:36:40,170] [    INFO] - Module ernie_gen_lover_words already installed in /home/aistudio/.paddlehub/modules/ernie_gen_lover_words[0m
    [33m[2020-08-25 23:36:48,536] [ WARNING] - use_gpu has been set False as you didn't set the environment variable CUDA_VISIBLE_DEVICES while using use_gpu=True[0m


    ['阿尔卑斯山，是我最想去的地方。', '阿尔卑斯山，是我最想去的地方，自从认识你，我的生活变的无限宽广，你是我永远的爱人!', '阿尔卑斯山，是我最想去的地方，自从认识你，我的生活变的无限宽广。', '阿尔卑斯山，是我最想去的地方，自从认识你，我的生活变的无限宽广，你是我永远的爱人。', '阿尔卑斯山，我也很想你，但我不下雨，愿意陪你淋雨。']


**完**
（希望小王可以收获爱情）
