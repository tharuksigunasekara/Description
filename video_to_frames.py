import os
from pytube import YouTube

def download_youtube_video(url, save_path='.'):
    """
    Downloads a YouTube video to the specified path.

    Parameters:
    - url: The URL of the YouTube video.
    - save_path: The directory to save the downloaded video. Defaults to the current directory.
    """
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    stream.download(output_path=save_path, filename="video.mp4")
    print(f"Downloaded '{yt.title}' to {save_path}/video.mp4")
    return os.path.join(save_path, "video.mp4")