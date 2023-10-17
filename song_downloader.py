from pytube import YouTube
from moviepy.editor import AudioFileClip
import os
import time

def create_progress_function():
    start_time = [time.time()]  # Using a single-item list to hold a mutable object
    total_size = [0]  # To store the total size of the file being downloaded

    def progress_function(stream, chunk, bytes_remaining):
        nonlocal total_size  # To access the outer function's variable
        if total_size[0] == 0:
            total_size[0] = stream.filesize  # Store total size on the first run

        # Current time
        current_time = time.time()

        # Check if it's the start of the download
        if bytes_remaining == total_size[0]:
            start_time[0] = current_time

        bytes_downloaded = total_size[0] - bytes_remaining
        elapsed_time = current_time - start_time[0]

        # Calculate download progress percentage
        progress = (bytes_downloaded / total_size[0]) * 100.0

        # Calculate the download speed
        download_speed = bytes_downloaded / elapsed_time if elapsed_time > 0 else 0
        download_speed_mbps = download_speed / 1_000_000 * 8  # Convert to Mbps

        # Print download speed and progress bar
        progress_bar_str = f"[{'=' * int(progress / 10)}{' ' * (10 - int(progress / 10))}]"  # Simple text-based progress bar
        print(
            f"\rDownload speed: {download_speed_mbps:.2f} Mbps | Progress: {progress:.1f}% {progress_bar_str}",
            end="",
        )

    return progress_function


def download_audio(youtube_url, base_output_path, folder_name):
    """
    Download audio from a YouTube video using pytube and save it to a specific folder.
    """
    # Define the path for the new folder.
    output_path = os.path.join(base_output_path, folder_name)

    # Create the folder if it doesn't exist.
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    try:
        progress_function = create_progress_function()
        yt = YouTube(youtube_url, on_progress_callback=progress_function)
        highest_quality_stream = yt.streams.filter(only_audio=True).first()

        if highest_quality_stream is not None:
            # Download the file with a temporary file name
            temp_file = highest_quality_stream.download(output_path=output_path, filename="temp")

            # Define the new file path (with .mp3 extension)
            new_file_path = os.path.join(output_path, "song.mp3")

            # If a file named "song.mp3" already exists, remove it to avoid conflict
            if os.path.exists(new_file_path):
                os.remove(new_file_path)

            # Convert the temporary file to MP3 format
            with AudioFileClip(temp_file) as audio:
                audio.write_audiofile(new_file_path, bitrate="320k")

            # Remove the original downloaded file (which is not in .mp3)
            os.remove(temp_file)

            print(f"\nDownload finished for {youtube_url}")
        else:
            print(f"No audio streams available for {youtube_url}")
    except Exception as e:
        print(f"An error occurred: {e}")


base_path = "./songs"
video_links = [
    "https://www.youtube.com/watch?v=kXHiIxx2atA",
    "https://www.youtube.com/watch?v=T0PIrzGW0fc",
    "https://www.youtube.com/watch?v=lsLskkCQ9qs",
    "https://www.youtube.com/watch?v=ppm5OacDijg",
    "https://www.youtube.com/watch?v=OrZLUuITgv8",
]
for i, link in enumerate(video_links, start=1):
    folder_name = str(i)
    download_audio(link, base_path, folder_name)
