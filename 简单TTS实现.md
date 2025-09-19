
```python
import torch
from transformers import AutoProcessor, AutoModelForSpeechSynthesis

# --- 选择一个预训练模型 ---
# 推荐几个中文模型，你可以根据需要选择：

# 1. 中文多说话人VITS模型 (声音自然，支持多种风格)
# model_name = "espnet/kan-bayashi_csmsc_vits" # 这是一个ESPnet的VITS模型，在Hugging Face上可用
# text_to_synthesize = "你好，欢迎来到文本转语音的世界！"

# 2. 中文FastSpeech2模型 (速度快，但可能需要单独的声码器，不过transformers通常会集成)
# model_name = "espnet/csmsc_tts_fastspeech2" # 另一个ESPnet的FastSpeech2模型
# text_to_synthesize = "深度学习让语音合成变得更加简单。"

# 3. 微软的中文多说话人SpeechT5模型 (质量非常好，支持情感、说话人风格迁移，但可能需要更多资源)
model_name = "microsoft/speecht5_tts"
text_to_synthesize = "这个模型能够将文字转换为逼真的语音。"

# SpeechT5需要一个说话人嵌入来模仿特定人的声音。
# 我们可以加载一个预设的说话人嵌入，或者从一段音频中提取。
# 这里我们用一个通用的说话人嵌入，让语音有一个默认的音色。
# 实际上你可以用 speecht5_tts/speaker_embeddings/cmu_arctic_speaker_embeddings.bin 这样的文件加载。
# 对于简单的例子，我们可以创建一个随机的嵌入，或者使用一个预设的平均嵌入
# 注意：随机嵌入会每次产生不同音色的声音，如果需要稳定音色，应加载预设的speaker_embedding
# from datasets import load_dataset
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0) # 这是一个预设的说话人嵌入

# 简化处理，直接加载一个固定的示例说话人嵌入（对于SpeechT5是必须的）
# 这个是 SpeechT5 官方文档中提供的示例嵌入，确保你已经下载了它。
# 如果没有，可以使用随机生成或者从其他音频中提取。
# 为了代码的可执行性，我们生成一个随机的（但每次运行音色会变）
# 实际生产中会使用固定的 xvector 嵌入文件
speaker_embeddings = torch.randn(1, 512) # 随机生成一个512维的说话人嵌入

print(f"正在加载模型: {model_name}...")

# 1. 加载处理器和模型
# processor 负责文本前端处理，如分词、G2P等
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForSpeechSynthesis.from_pretrained(model_name)

# 确保模型在 GPU 上（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("模型加载完成。开始合成语音...")

# 2. 准备输入：文本转为模型可识别的输入 ID
# 对于 SpeechT5，输入 IDs 还需要 speaker_embeddings
inputs = processor(text=text_to_synthesize, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 3. 生成语音波形
# generate 方法会自动调用内部的声码器
with torch.no_grad():
    speech = model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings).to("cpu").numpy()

print("语音合成完成。")

# 4. 保存或播放语音
# 你可以使用 torchaudio 或 scipy.io.wavfile 来保存。
# 如果你想直接播放，可以在Jupyter Notebook或Colab中使用 IPython.display
import soundfile as sf
output_filename = "synthesized_speech.wav"
sf.write(output_filename, speech.squeeze(), samplerate=model.config.sampling_rate)

print(f"语音已保存到 {output_filename}")

# 可选：在 Jupyter/Colab 中直接播放
# from IPython.display import Audio
# Audio(speech.squeeze(), rate=model.config.sampling_rate)
