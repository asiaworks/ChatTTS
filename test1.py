import torch
import ChatTTS
from IPython.display import Audio
import soundfile

# 初始化ChatTTS
chat = ChatTTS.Chat()
#chat.download_models()
chat.load()

# 定义要转换为语音的文本
#texts = ["你好，欢迎使用ChatTTS！"]

# 生成语音
#wavs = chat.infer(texts, use_decoder=True)

#params_infer_code = {'prompt':'[speed_5]', 'temperature':.3}
#params_refine_text = {'prompt':'[oral_2][laugh_0][break_6]'}

#wav = chat.infer('四川美食可多了，有麻辣火锅、宫保鸡丁、麻婆豆腐、担担面、回锅肉、夫妻肺片等，每样都让人垂涎三尺。')
#Audio(wav[0], rate=24_000, autoplay=True)
#torchaudio.save("/workspaces/ChatTTS/output2.wav", torch.from_numpy(wav[0]), 24000)
#soundfile.write("output1.wav", wav[0], 24000)

inputs_en = """
四川美食可多了，有麻辣火锅、宫保鸡丁、麻婆豆腐、担担[uv_break]面、回锅[laugh]肉、夫妻肺片等，每样都让人垂涎三尺。
[uv_break]it supports mixed language input [uv_break]and offers multi speaker
capabilities with precise control over prosodic elements like
[uv_break]laughter[uv_break][laugh], [uv_break]pauses, [uv_break]and intonation.
""".replace('\n', '') # English is still experimental.

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_4]',
)
#rand_spk = chat.sample_random_speaker()
#print(rand_spk) # save it for later timbre recovery

# params_infer_code = ChatTTS.Chat.InferCodeParams(
#     spk_emb = rand_spk, # add sampled speaker 
#     temperature = .3,   # using custom temperature
#     top_P = 0.7,        # top P decode
#     top_K = 20,         # top K decode
# )

audio_array_en = chat.infer(
    inputs_en,
    params_refine_text=params_refine_text)
soundfile.write("output1.wav", audio_array_en[0], 24000)

