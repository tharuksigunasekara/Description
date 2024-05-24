import os
import cv2
import googleapiclient.discovery
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from tqdm import tqdm

import os

from data_processing_pipeline import preprocess_text
from video_to_audio import convert_video_to_audio_ffmpeg, wav_to_mono_flac
from video_to_frames import download_youtube_video
from vision.object.test import predict_yolo
from vision.scene.run_placesCNN_unified import process_image

import os
from collections import defaultdict, Counter

# API information
api_service_name = "youtube"
api_version = "v3"
DEVELOPER_KEY = 'AIzaSyDgCxRsHNBYqRdB8s3K8W-dTLEzrdxRBu8'
# API client
youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)


def get_video_info(video_id):
    request = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    )
    response = request.execute()
    return response


def extract_video_info(data):
    # Check if 'items' is in data and has at least one item
    if not data.get('items') or len(data['items']) == 0:
        return "No video information available."

    video_info = data['items'][0]
    extracted_info = {
        "title": video_info['snippet']['title'],
        "description": video_info['snippet']['description'],
        "likeCount": video_info['statistics'].get('likeCount', "Not available"),
        "commentCount": video_info['statistics'].get('commentCount', "Not available")
    }
    return extracted_info


def video_to_frames(video_path, frames_dir, skip_frames=0):
    """
    Extracts frames from a video file and saves them as individual image files in a specified directory.

    Parameters:
        video_path (str): The path to the video file.
        frames_dir (str): The directory to save the extracted frames.
        skip_frames (int): Number of frames to skip after saving one.
    """
    if os.path.exists(frames_dir):
        # Delete all files in the directory
        for file in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    # Make directory to save frames, if it doesn't exist
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    print("Extracting frames...")
    # Load the video
    video = cv2.VideoCapture(video_path)
    count = 0
    frame_index = 0

    while True:
        # Read video frame by frame
        success, frame = video.read()
        if not success:
            break  # When no more frames, exit loop

        # Only save frames according to the skip parameter
        if frame_index % (skip_frames + 1) == 0:
            # Save each frame to the directory
            frame_filename = os.path.join(frames_dir, f"frame_{count:05d}.png")
            cv2.imwrite(frame_filename, frame)
            count += 1

        frame_index += 1

    video.release()  # Release the video object
    print(f"Extracted {count} frames and saved to {frames_dir}")


def process_folder(folder_path):
    # Dictionaries to hold aggregated data
    object_counts = Counter()
    environment_types = Counter()
    scene_categories = defaultdict(float)
    scene_attributes = Counter()

    # Get a list of all PNG files in the directory
    files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    # Process each file in the directory with a progress bar
    for filename in tqdm(files, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)

        # Predict objects in the image
        predicted_yolo = predict_yolo(image_path)
        # Process scene information
        predicted_scene = process_image(image_path)

        # Update object counts
        for object_type, count in predicted_yolo.items():
            object_counts[object_type] += count

        # Update environment type
        environment_types[predicted_scene['type_of_environment']] += 1

        # Update scene categories with weighted averages
        for weight, category in predicted_scene['scene_categories']:
            scene_categories[category] += weight

        # Update scene attributes
        for attribute in predicted_scene['scene_attributes']:
            scene_attributes[attribute] += 1

    # Normalize scene category weights by number of images processed
    total_images = len(files)
    if total_images > 0:
        for category in scene_categories:
            scene_categories[category] /= total_images

    return {
        'object_counts': object_counts,
        'environment_types': environment_types,
        'scene_categories': dict(scene_categories),
        'scene_attributes': scene_attributes
    }


def setup_run_folder(base_path='runs'):
    # Ensure the base runs folder exists
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    # Find the highest numbered existing run folder
    last_run = -1
    for folder in os.listdir(base_path):
        if folder.startswith('run_') and folder[4:].isdigit():
            last_run = max(last_run, int(folder[4:]))

    # Create the next run folder
    new_run_folder = f'run_{last_run + 1}'
    new_run_path = os.path.join(base_path, new_run_folder)
    os.mkdir(new_run_path)

    # Create a log file in the new run folder
    log_file_path = os.path.join(new_run_path, 'log.txt')
    with open(log_file_path, 'w') as log_file:
        log_file.write('Log started for ' + new_run_folder + '\n')

    return new_run_path + "/", log_file_path


from sklearn.feature_extraction.text import TfidfVectorizer


def extract_keywords_tfidf(doc, mode="threshold", top_n=10, threshold=0.03):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([doc])
    feature_names = vectorizer.get_feature_names_out()

    # Get the top keywords for the document
    tfidf_scores = tfidf_matrix[0]
    sorted_items = sorted(zip(tfidf_scores.indices, tfidf_scores.data), key=lambda x: x[1], reverse=True)
    if mode == "threshold":
        print("Threshold mode")
        top_keywords = [(feature_names[idx], score) for idx, score in sorted_items if score > threshold]
    else:
        top_keywords = [(feature_names[idx], score) for idx, score in sorted_items[:top_n]]

    return top_keywords

def log_txt(log_file_path, text):
    with open(log_file_path, 'a') as log_file:
        log_file.write(text + '\n')

if __name__ == '__main__':
    # Execute the function
    new_run_path, log_file_path = setup_run_folder()
    # new_run_path, log_file_path = "runs/run_8/", "runs/run_8/log.txt"
    print(f'New run folder created: {new_run_path}')
    print(f'Log file created: {log_file_path}')

    video_id = "fLeJJPxua3E"  # Replace YOUR_VIDEO_ID with the actual video ID
    video_info = get_video_info(video_id)
    extracted_info = extract_video_info(video_info)

    # Apply preprocess_text
    text = extracted_info["description"]
    text = preprocess_text(text)
    print(text, "\n\n\n")

    yt_path = download_youtube_video(f"https://www.youtube.com/watch?v={video_id}", save_path=new_run_path)

    frames_dir = new_run_path + "/frames"
    video_to_frames(new_run_path + "video.mp4", frames_dir, 30)

    results = process_folder(frames_dir)
    print(results, "\n\n\n")

    # yt_path = r"C:\D\Projects\Mora\Youtube Part 1\video.mp4"
    # yt_path = r"runs\run_8/video.mp4"
    print(yt_path)
    # print(yt_path)
    # print(yt_path)
    sound = AudioSegment.from_file(new_run_path + "video.mp4", "mp4")
    sound.export(f"{new_run_path}/audio.wav", format="wav")

    # video_clip = VideoFileClip("video.mp4")
    # audio_clip = video_clip.audio
    # audio_clip.write_audiofile("audio.mp3")
    # video_clip.close()
    # audio_clip.close()
    #
    # audio = AudioSegment.from_mp3("audio.mp3")
    # wav_file_path = 'audio.wav'
    # audio.export(wav_file_path, format='wav')

    # exit()

    text = "get motivated get inspired video greatest motivation inspiration one best speaker time eric thomas best motivational speech posted channel inspire motivate listen everyday motivation achieve dream successful video meant motivate inspire need extra motivation life look motivation instrumental taking first step toward dream many lack motivation inspiration due tv programming people surround job etc today choose success hope motivational video aide journey destiny thank speaker eric thomas blueprint success way stay connected motiversity stay motivated subscribe new motivational video every week download top quote time join newsletter exclusive update discount deal read weekly blog shop official motivational canvas apparel become member loyal community follow motiversity social medium find u everywhere discord facebook instagram tiktok website follow motiversity music podcast platform spotify music apple music motivation daily podcast mindset app follow motiversity youtube channel submit motiversity speech music footage new motiversity motivational canvas art join motivational list get exclusive video discount update recommended reading list amazing author like brendan burchard david goggins james clear dale carnegie stephen r covey nick winter tara westover mel robbins steven pressfield charles duhigg elizabeth gilbert david allen billy alsbooks walter bond kevin kruse zac bissonnette disclaimer please note receive commission amazon use referral link thank support fair use disclaimer copyright disclaimer section copyright act allowance made fair use purpose criticism commenting news reporting teaching scholarship research fair use use permitted copyright statute might otherwise infringing nonprofit educational personal use tip balance favor fair use purpose making motivational video steal people video make quality educational motivational video version share viewer motiversity right video clip accordance fair use repurposed intent educating inspiring others legal content owner video posted channel would like removed please message help u caption translate video help u caption translate video motiversity inspiration motivation"

    # image_data = {'object_counts': Counter({'person': 73, 'car': 5, 'tie': 3, 'laptop': 2, 'snowboard': 1}),
    #               'environment_types': Counter({'indoor': 53, 'outdoor': 13}),
    #               'scene_categories': {'catacomb': 0.007895504491347256, 'stage/indoor': 0.022657310110375736,
    #                                    'movie_theater/indoor': 0.028447121946197567, 'sky': 0.0034662734322024116,
    #                                    'grotto': 0.004046076287825902, 'television_studio': 0.022337851763674706,
    #                                    'music_studio': 0.025739661502567204, 'coffee_shop': 0.0018532116135412996,
    #                                    'home_theater': 0.03783864014302239, 'living_room': 0.0016891676368135395,
    #                                    'waiting_room': 0.000924794003367424, 'hotel_room': 0.0055953242787809086,
    #                                    'airplane_cabin': 0.003614177250049331, 'berth': 0.0009994476356289604,
    #                                    'bar': 0.01406184888698838, 'airport_terminal': 0.003263330566837932,
    #                                    'science_museum': 0.006953229224591544, 'server_room': 0.003437217550747322,
    #                                    'pub/indoor': 0.020246813920411198, 'discotheque': 0.008437824971748121,
    #                                    'beer_hall': 0.0053789020149093685, 'arena/performance': 0.028518140062012455,
    #                                    'locker_room': 0.0006825311504530184, 'conference_center': 0.02732069067882769,
    #                                    'bowling_alley': 0.008886487082098469, 'gymnasium/indoor': 0.02351636134765365,
    #                                    'martial_arts_gym': 0.002414636539690422, 'beauty_salon': 0.009750316653287771,
    #                                    'lecture_room': 0.007128328423608433, 'classroom': 0.0023471035740592265,
    #                                    'car_interior': 0.01037998271710945, 'cockpit': 0.0015116278646570263,
    #                                    'auto_showroom': 0.0021605281949494824, 'parking_lot': 0.00032442875883796,
    #                                    'orchestra_pit': 0.001173545578212449, 'raceway': 0.0018304421030210726,
    #                                    'parking_garage/indoor': 0.0012585174179438388,
    #                                    'train_station/platform': 0.0024082393695910773,
    #                                    'subway_station/platform': 0.010519878148581043,
    #                                    'booth/indoor': 0.0019226954741911454, 'clean_room': 0.0011399450401465099,
    #                                    'jail_cell': 0.027320764168645397, 'elevator/door': 0.006774410548986811,
    #                                    'bow_window/indoor': 0.0006237007451779915, 'archive': 0.004147258671847257,
    #                                    'pharmacy': 0.0022798613504026875, 'bookstore': 0.001275229070222739,
    #                                    'fire_escape': 0.0007481475106694481, 'restaurant': 0.0008420900752147039,
    #                                    'restaurant_kitchen': 0.001059247124375719,
    #                                    'elevator_lobby': 0.0007852933962236751, 'army_base': 0.011575461393504433,
    #                                    'igloo': 0.0031599411599789605, 'street': 0.0038616003637964077,
    #                                    'cemetery': 0.0022726629042264185, 'crosswalk': 0.0014569126069545746,
    #                                    'ski_slope': 0.001346807316594729, 'construction_site': 0.001939248124306852,
    #                                    'rope_bridge': 0.0035805282832095118, 'stage/outdoor': 0.0031637678092176266,
    #                                    'playground': 0.0007292889058589935, 'bridge': 0.0006434296568234762,
    #                                    'ice_skating_rink/outdoor': 0.016930933646631962,
    #                                    'promenade': 0.0005450033667412671, 'ocean': 0.00047687768484606886,
    #                                    'ice_floe': 0.002835213968699629, 'ice_shelf': 0.005092589071754253,
    #                                    'lake/natural': 0.0004936033351854844, 'snowfield': 0.0015643373921965108,
    #                                    'ice_skating_rink/indoor': 4.0396014369572654e-05,
    #                                    'glacier': 0.0034360049123113804, 'crevasse': 0.004188045125567552,
    #                                    'mountain_snowy': 0.0012718932420918436, 'elevator_shaft': 0.006287296725945039,
    #                                    'veterinarians_office': 0.01840046454559673,
    #                                    'hospital_room': 0.003132618512168075,
    #                                    'natural_history_museum': 0.011098281113487301, 'bathroom': 0.001227455044334585,
    #                                    'aquarium': 0.002270879506161719, 'operating_room': 0.003217861677209536,
    #                                    'sauna': 0.002236746928908608, 'shower': 0.00210539951468959,
    #                                    'alley': 0.011743562296032906, 'arcade': 0.0024757331067865544,
    #                                    'medina': 0.0017536139172134976, 'corridor': 0.002586820191054633,
    #                                    'basement': 0.00034385434154308206, 'stable': 0.0003411403828949639,
    #                                    'iceberg': 0.004191145752415512, 'campsite': 0.000812374111829382,
    #                                    'railroad_track': 0.011325176015044704, 'burial_chamber': 0.0007357439308455496,
    #                                    'butchers_shop': 0.00038234452067902595, 'slum': 0.0004890412656646786,
    #                                    'amphitheater': 0.0002675206598007318}, 'scene_attributes': Counter(
    #         {'man-made': 66, 'no horizon': 64, 'enclosed area': 55, 'indoor lighting': 50, 'cloth': 49, 'working': 36,
    #          'stressful': 25, 'natural light': 24, 'congregating': 22, 'open area': 20, 'glossy': 19, 'metal': 16,
    #          'competing': 16, 'vertical components': 14, 'spectating': 11, 'natural': 11, 'scary': 10, 'dry': 10,
    #          'glass': 9, 'socializing': 9, 'transporting': 8, 'far-away horizon': 8, 'cold': 8, 'sunny': 8, 'matte': 7,
    #          'sports': 7, 'clouds': 7, 'trees': 6, 'snow': 6, 'wood': 5, 'medical activity': 5, 'aged': 5, 'ocean': 4,
    #          'reading': 4, 'training': 4, 'warm': 4, 'ice': 4, 'horizontal components': 3, 'boating': 2,
    #          'waiting in line': 2, 'exercise': 2, 'asphalt': 2, 'driving': 2, 'sterile': 2, 'railroad': 2, 'carpet': 1,
    #          'soothing': 1, 'paper': 1, 'biking': 1, 'foliage': 1, 'rugged scene': 1, 'rusty': 1})}

    image_data = results

    extract_keywords = extract_keywords_tfidf(text, "threshold")

    # Combine image data into a single dictionary for easier processing
    image_terms = {}
    image_terms.update(image_data['object_counts'])
    image_terms.update({k: v for k, v in image_data['scene_categories'].items()})  # Convert categories to dictionary
    image_terms.update(image_data['scene_attributes'])

    def calculate_similarity(extracted_terms, image_terms):
        # log the extracted terms
        log_txt(log_file_path, "Extracted terms:")
        for term in extract_keywords:
            log_txt(log_file_path, f"{term}")

        # log the image terms
        log_txt(log_file_path, "\nImage terms:")
        for term in image_terms:
            log_txt(log_file_path, f"{term}: {image_terms[term]}")

        # Create sets of keywords
        extracted_set = {term[0] for term in extracted_terms}
        image_set = set(image_terms.keys())

        # Find common elements
        common_terms = extracted_set.intersection(image_set)

        # Score based on the number of common terms and their weights in the image data
        score = sum(image_terms[term] for term in common_terms if term in image_terms)

        # Normalize the score by the maximum possible score
        max_score = sum(image_terms.values())
        return score / max_score if max_score != 0 else 0


    # Convert extracted_keywords to a dictionary for easy processing
    extracted_terms = {term[0]: term[1] for term in extract_keywords}

    # Calculate similarity score
    similarity_score = calculate_similarity(extracted_terms, image_terms)
    print(f"Similarity Score: {similarity_score:.2f}")
