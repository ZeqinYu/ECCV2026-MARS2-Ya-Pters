# 🏆 MARS2 2026 Challenge on Multimodal Reasoning: From Multimodal Perception to Complex Reasoning

1st Place Solution for the [Video Temporal Grounding (VTG) Track](https://eval.ai/web/challenges/challenge-page/2698/overview) of the [MARS2 2026 Workshop at ECCV 2026](https://mars2workshop.github.io/eccv2026/) by the **"Ya PTers"** team.

Team members: Zeqin Yu<sup>1,5</sup>, Zhixuan Wu<sup>2,5</sup>, Peiyu Zhuang<sup>1</sup>, and Yang Xu<sup>3,4</sup>, under the supervision of Prof. Jiangqun Ni<sup>1</sup>.

<sup>1</sup> *Sun Yat-sen University*  
<sup>2</sup> *Beijing University of Posts and Telecommunications*  
<sup>3</sup> *Center for Strategic Studies, Chinese Academy of Engineering*  
<sup>4</sup> *Tsinghua University*  
<sup>5</sup> *Nanyang Technological University*  

## Method

We propose **TempoFlex**, a duration-adaptive inference framework for Video Temporal Grounding. TempoFlex dynamically allocates the visual-token budget according to video duration, enabling longer videos to retain more visual information while keeping the inference cost bounded. Our final solution uses 5 FPS video sampling, an adaptive visual-token budget ranging from 11,264 to 18,432 tokens, and timestamp normalization and boundary correction for reliable output generation. TaRO-8B is used as the base video-language model.

## Usage

### 1. Environment

```bash
conda env create -f taro_sft_environment.yml
conda activate taro_sft
```

### 2. Model

Download the required checkpoint:

```bash
git lfs install
git clone https://huggingface.co/zhengmh/TaRO-8B ./weights/TaRO-8B
```

### 3. Inference

```bash
python3 TempoFlex.py \
  --model-path ./weights/TaRO-8B \
  --input-jsonl /path/to/VTG_QA.jsonl \
  --video-dir /path/to/mars2_videos \
  --output-dir ./outputs/tempoflex \
  --gpu-id 0
```

The final prediction file will be saved to:

```text
outputs/tempoflex/out.jsonl
```

## Acknowledgements

We thank the authors of [TaRO](https://github.com/oceanflowlab/TaRO) for making their model and code publicly available.

