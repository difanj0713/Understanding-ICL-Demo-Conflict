---
license: mit
task_categories:
- image-to-text
- text-to-image
tags:
- In-context learning
- ICL
- Multimodal
- Vision-Language
- VLLMs
size_categories:
- 1K<n<10K
---
# VL-ICL Bench
VL-ICL Bench: The Devil in the Details of Benchmarking Multimodal In-Context Learning

[[Webpage]](https://ys-zong.github.io/VL-ICL/) [[Paper]](https://arxiv.org/abs/2403.13164) [[Code]](https://github.com/ys-zong/VL-ICL)


## Image-to-Text Tasks

In all image-to-text tasks `image` is a list of image paths (typically one item - for interleaved cases there are two items).

### Fast Open-Ended MiniImageNet
Frozen introduces the task of fast concept binding for MiniImageNet. The benchmark has a fixed structure so only the given support examples can be used for a given query example. We store all support images in the `support` directory and all query images in the `query` directory. We provide a `support.json` file with information about the support images, but these do not need to be used. Because of the fixed structure of the benchmark, all needed information is stored inside `query.json` file. This file includes information about the query image, the list of artificial `classes` that can be used for constructing the task with the given query image, as well as five examples for each class (we store the image paths and the caption that refers to all these examples). We used the 5-way 5-shot setting, but we are free to take only the query example class and between one and four other classes. For our experiments we use a 2-way setting. For each class we can take up to 5 support examples. We have 200 query examples and total of 5000 support examples, but we can extend it for up to 2500 query examples with the corresponding number of support examples.

Source of data: https://fh295.github.io/frozen.html

### CLEVR Count Induction
We repurpose the CLEVR dataset to construct tasks where we try to count the number of objects with a given characteristic, for example all large objects. The available attributes are shape, size, material and colour. The specified criterion is included within the `question`, for example `shape: large`, and the count itself is in the `answer`. We have 800 images in the support set and 200 in the query set.

Source of data: https://cs.stanford.edu/people/jcjohns/clevr/

### Operator Induction
The goal of this task is to predict what is the result. There is text in the image saying `A ? B`, where A and B are digits between 0 and 9. We randomly split all available options into 80 support and 60 query examples. For constructing the tasks we sample the images completely randomly, we sample the operation which `?` represents, and then take the corresponding answer. We store 3 answers for each example in a list for the support examples: `[A+B, A-B, AxB]`, and the result can be accessed with the appropriate index. The `question` that we ask is always `What is the result of the following mathematical expression?`. We generated the images using PIL library, using Arial font with size 100 on images of size 256x256. We store the `operator` for each query example, and we have 20 examples for each operator.

### Interleaved Operator Induction
We also include an alternative interleaved version of operator induction where we input the two digits as separate images. The `question` that we ask is `What is the result of the following mathematical expression?`.

### TextOCR
In TextOCR the goal is to recognize the text that is shown in the red rectangle. In our version of TextOCR there is always only one red rectangle in an image. We take the original training set for setting aside 800 support examples and the validation set for 200 query examples. We use the largest text in the image to simplify the task, and we make sure to filter out all cases that are not valid (marked as `.` in the annotation). We also filter out the rotated images. The `question` asked is `What text is shown in the red box?` and the answer is the text itself. We maintain various metadata, including the image and annotation id, width, height, box coordinates, points for the text, overall area.

Source of data: https://textvqa.org/textocr/

### MiniImageNet Matching
In this variation of MiniImageNet we try to predict if two examples are from the same class. We have 400 query pairs and 1600 support pairs, evenly distributed between same and different classes. Each support pair includes a pair of examples from the same class and a pair of examples from different classes. The `question` is always `Do the two images satisfy the induced relationship?` and the `answer` is either `Yes` or `No`. We used our earlier Fast Open-Ended MiniImageNet to create this matching dataset.

Source of data: https://fh295.github.io/frozen.html


## Text-to-Image Tasks

### Fast Open-Ended T2I MiniImageNet
We introduce a variation of Fast Open-Ended MiniImageNet where the goal is to generate an image of the imaginary class as given by the support examples. The details are similar to our other version of Fast Open-Ended MiniImageNet, but the question is instead `Generate a ` followed by the name of the imaginary class. We store the imaginary class in `task_label` field, and the real-world label in `answer` for the query examples (the support set examples have there the imaginary class). The labels were obtained from the real-world version of the benchmark. These labels can be used to assess if the generated image represents the desired imaginary class.

Source of data: https://fh295.github.io/frozen.html

### CoBSAT
We reuse the CoBSAT benchmark for few-shot image generation tasks. We have 800 support and 200 query examples, and these are organized in such a way that for each of the 100 scenarios (defined by the task -- e.g. colour, and the choice of the latent variable -- e.g. object value), we have 8 support and 2 query examples. When sampling the support examples, we need to ensure that these share the same `task` and value of the latent variable `latent`, which can be either the value of `attribute` or `object`. The `question` has the value of the latent variable and defines what image should be generated. The `image` is the generated image. The `answer` is a list [value of the latent variable, value of the non-latent variable]. For each image we also store the values of the `object`, `attribute`. 

Source of data: https://github.com/UW-Madison-Lee-Lab/CoBSAT


## Text ICL Variations

We have also released the text variations of CLEVR, Operator Induction, and interleaved Operator Induction datasets to reproduce the comparison of multimodal and text ICL (Figure 7). You can either use the `query.json` in `{dataset}_text/` folder for "text support set + text query", or use the `query.json` in `{dataset}/` folder for "text support set + multimodal query".