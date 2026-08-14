<p align="center">
  <img src="asserts/apollo-logo.png" alt="Apollo logo" width="150"/>
</p>

# Apollo 繁體中文導讀

<p align="center">
  <a href="README.md">English README</a> |
  <a href="https://arxiv.org/abs/2409.08514">Paper</a> |
  <a href="https://cslikai.cn/Apollo/">Demo</a>
</p>

> 為方便臺灣及其他使用繁體中文的讀者快速認識 Apollo，也希望讓更多人有機會接觸這個 open-source project，我們冒昧補上一份簡短導讀。若這項補充與專案目前的文件規劃不一致，敬請維護者包涵，也歡迎隨時指正。完整技術內容與最新資訊仍以 [English README](README.md) 及原始 paper 為準。

## Project overview

Apollo 是一個用於 high-quality audio restoration 的 generative model，目標是從 compressed audio 重建較完整的音訊內容。模型透過 explicit **frequency band split module** 建立不同 frequency bands 之間的關係，在保留 low-frequency information 的同時重建 mid- and high-frequency content。

研究使用 MUSDB18-HQ 與 MoisesDB 進行評估。模型架構、實驗設定與完整結果請參閱 [paper](https://arxiv.org/abs/2409.08514)；此導讀不取代原始研究說明。

## Quick links

- [English README](README.md)
- [Apple Silicon macOS notes](MACOS_ARM64.md)
- [Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess)
- [Hugging Face checkpoint](https://huggingface.co/JusperLee/Apollo)
- [Demo](https://cslikai.cn/Apollo/)

## Installation

官方環境安裝方式：

```bash
git clone https://github.com/JusperLee/Apollo.git && cd Apollo
conda create --name look2hear --file look2hear.yml
conda activate look2hear
```

Apple Silicon 使用者可另參閱 [MACOS_ARM64.md](MACOS_ARM64.md)，其中記錄 native arm64 Python、PyTorch MPS 與目前已知限制。

## Inference

以 repository 內的 public fixture 執行：

```bash
python inference.py \
  --in_wav=asserts/input_wav.wav \
  --out_wav=output.wav \
  --device=auto
```

`--device=auto` 依序選擇 CUDA、Apple Silicon MPS、CPU。預設 checkpoint 來自官方 `JusperLee/Apollo` Hugging Face repository；自訂 `--checkpoint` 只接受明確存在且使用者信任的 local file。

可明確指定 device：

```bash
python inference.py --in_wav=input.wav --out_wav=output.wav --device=mps
python inference.py --in_wav=input.wav --out_wav=output.wav --device=cpu
```

請勿以技術測試通過推論 audio quality 已完成主觀驗收；輸出結果仍應由使用者自行聆聽與判斷。

## Dataset and training

Apollo 使用 MUSDB18-HQ 與 MoisesDB。dataset 取得方式、Source Activity Detection、data augmentation、MP3 codec simulation、rescaling 與 HDF5 preprocessing 的完整說明，請以 [English README](README.md) 和 [Apollo-data-preprocess](https://github.com/JusperLee/Apollo-data-preprocess) 為準。

Training command：

```bash
python train.py --conf_dir=configs/apollo.yml
```

## Results, license, and citation

- Results figures: [bitrates](asserts/bitrates.png), [music types](asserts/types.png)
- License: [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/)
- Citation information: [English README](README.md#citation)
- Contact: `tsinghua.kaili@gmail.com`

感謝 Apollo maintainers 開放原始碼與研究成果。這份導讀僅協助繁體中文讀者找到正確入口；若內容與 upstream 有差異，請一律以上游英文文件為準。
