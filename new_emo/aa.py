from pydub import AudioSegment
import whisper
import ffmpeg
import shap
from transformers import pipeline


# Load the audio file
audio_file = AudioSegment.from_file("graduation.m4a", format="m4a")

# Set the start and end time for the segment (in milliseconds)
start_time = 0
end_time = 2000
for i, audio_end in enumerate(range(2000, len(audio_file), 2000)):
    #print(audio_file) 
    print(audio_end)
    segment = audio_file[0:audio_end]
    segment.export("segment" + str(i) + ".mp3", format="mp3")

classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)

model = whisper.load_model("base", device="cuda")
_ = model.half()
_ = model.cuda()

for m in model.modules():
    if isinstance(m, whisper.model.LayerNorm):
        m.float()


from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

videos = []
for i in range(1,i):
    path = "segment" + str(i-1) + ".mp3"
    old_path = "segment" + str(i) + ".mp3"

    result = model.transcribe(path, verbose=True, language='japanese').get("text")
    pre_result = model.transcribe(old_path, verbose=True, language='japanese').get("text")
    new_emos = prediction = classifier(result)[0]
    old_emos = prediction = classifier(pre_result)[0]
    
    emos = []
    #print(result)
    for now, pre in zip(new_emos,old_emos):
        emos.append(now.get("score") - pre.get("score"))
    print(emos.index(max(emos)))
    if max(emos) < 0.01:
        videos.append(VideoFileClip("normal.MOV").subclip(0,2))
    else:
        videos.append(VideoFileClip("surprise.MOV").subclip(0,2))


audio_file = AudioFileClip("graduation.m4a")
final_clip = concatenate_videoclips(videos)
final_clip.set_audio("graduation.m4a")
final_clip.write_videofile("result.mp4")

    
