
from pytube import YouTube
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import re
import speech_recognition as sr
from pydub import AudioSegment

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

try:
    video_url = 'https://www.youtube.com/watch?v=K1FxeAGmoo0'
    output_path = '.'

    youtube = YouTube(video_url)
    video_stream = youtube.streams.get_highest_resolution()
    original_title = video_stream.title
    video_path = f"{output_path}/{original_title}.mp4"
    video_path = video_stream.download(output_path)
    new_title = "Renamed_" + sanitize_filename(original_title)
    renamed_video_path = f"{output_path}/{new_title}.mp4"
    os.rename(video_path, renamed_video_path)
    output_audio_path = f"{output_path}/{new_title}.mp3"
    video_clip = VideoFileClip(renamed_video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path)
    video_clip.close()
    audio_clip.close()

    audio = AudioSegment.from_mp3(output_audio_path)
    wav_file_path = output_audio_path.replace('.mp3', '.wav')
    audio.export(wav_file_path, format='wav')
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
    audio_text = recognizer.recognize_google(audio_data=audio_data)
    os.remove(wav_file_path)
    print(f"Transcription:\n{audio_text}")

except Exception as e:
    print(f"An error occurred: {str(e)}")