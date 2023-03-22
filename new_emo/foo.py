from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip


video1 = VideoFileClip("anger.mp4").subclip(0,2)

audio = AudioFileClip("audio.m4a")
#VideoFileClip(".mp4")
