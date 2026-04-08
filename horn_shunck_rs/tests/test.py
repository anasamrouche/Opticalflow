import os
import horn_schunck_rs
import numpy as np
import rich as r
import cv2
from typing import Dict, List
from time import time

video_paths = {
    "pingpongsd": "./tests/pingpongsd.mp4",
    "pingpong": "./tests/pingpong.mp4",
    "pingponghd": "./tests/pingponghd.mp4"
}

parameters: Dict["str", List[float|int]] = {
    "alphas": [0.001],
    "steps": [5e-5],
    "iteration_limits": [50, 500]
}

frameCount = 80

def generate_videos_by_gradient(video_set: Dict[str, str]):
    for video_name, path in video_set.items():
        r.print(f"Traitement du fichier: {path}")
        """
        Ouverture du fichier vidéo et enregistrement des paramètres
        """
        video = cv2.VideoCapture(path)
        frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))

        """
        Représentation de la vidéo sous la forme d'un array numpy
        Il y a peut-être une méthode appelable pour s'en charger plutôt que de passer par une boucle for
        Il faudrait que je me renseigne
        """
        temp_buffer = np.empty((frameCount, frameHeight, frameWidth), np.float32)
        video_buffer = np.empty((frameCount, frameHeight, frameWidth), np.float32)

        count = 0
        keep_going = True
        while count < frameCount and keep_going:
            keep_going, new_frame = video.read()
            temp_buffer[count] = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
            count += 1

        video_buffer = temp_buffer
        r.print(f"Vidéo de taille {video_buffer.shape[1:]} et de longueur {video_buffer.shape[0]}")
        """
        Résolution par la méthode du gradient
        Et écriture de la vidéo obtenue
        """
        for alpha_squared in parameters["alphas"]:
            for step in parameters["steps"]:
                for MaxIter in parameters["iteration_limits"]:
                    counts = np.zeros(MaxIter) #type: ignore
                    match video_name:
                        case "pingpongsd": output_name = f"tests/Norm_L1/gradient_results/low_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                        case "pingpong": output_name = f"tests/Norm_L1/gradient_results/standard_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                        case _: output_name = f"tests/Norm_L1/gradient_results/high_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                    if not os.path.exists(os.path.split(output_name)[0]):
                        os.makedirs(os.path.split(output_name)[0])
                    r.print(f"Résolution par gradient avec paramètres alpha squared : {alpha_squared}, pas: {step}, itérations max : {MaxIter}")
                    optical_flow_x, optical_flow_y, counts  = horn_schunck_rs.solve_gradient_descent(video_buffer, alpha_squared, step, MaxIter, 1e-3, False)
                    output = cv2.VideoWriter(f"{output_name}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frameWidth, frameHeight), isColor=False) #type: ignore

                    for frame_x, frame_y in zip(optical_flow_x, optical_flow_y):
                        movement_detection = (255 * ((frame_x ** 2 + frame_y**2) > 0.1)).astype(np.uint8)
                        output.write(movement_detection)
                    output.release()
                    r.print(counts)
                    r.print(f"Fichier {output_name} écrit.", end="\n\n")

generate_videos_by_gradient(video_paths)

# def lucas_kanade_sparse(video_path, max_corners=100):
#     """
#     Flot optique sparse avec Lucas-Kanade.
#     Suit des points d"interet entre frames successives.
#     """
#     cap = cv2.VideoCapture(video_path)
#     # Parametres pour la detection de coins
#     feature_params = dict(
#     maxCorners=max_corners,
#     qualityLevel=0.3,
#     minDistance=7,
#     blockSize=7
#     )
#     # Parametres Lucas-Kanade
#     lk_params = dict(
#     winSize=(15, 15),
#     maxLevel=2,
#     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#     )
#     ret, old_frame = cap.read()
#     old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#     p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#     trajectories = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Calcul du flot optique
#         p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#         # Selection des bons points
#         if p1 is not None:
#             good_new = p1[status == 1]
#             good_old = p0[status == 1]
#             # Stockage des deplacements
#             for new, old in zip(good_new, good_old):
#                 trajectories.append({
#                     "old": old.flatten(),
#                     "new": new.flatten(),
#                     "flow": (new - old).flatten()
#                 })
#         old_gray = frame_gray.copy()
#         p0 = good_new.reshape(-1, 1, 2)
    
#     cap.release()
#     return trajectories

# def lucas_kanade_benchmark(video_set: Dict[str, str]):
#     for video_quality, video_path in video_set.items():
#         time1 = time()
        