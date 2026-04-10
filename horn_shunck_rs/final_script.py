import os
import horn_schunck_rs
import numpy as np
from numpy import ndarray
import rich as r
import cv2
from typing import Dict, List, Tuple
import timeit

gradient_parameters: Dict[str, List[float|int]] = {
    "alphas": [1e-3, 1e-2, 1e-1],
    "iteration_limits": [30],
    "steps": [0.5, 1, 5],
}

GS_parameters: Dict[str, List[float|int]] = {
    "alphas": [1e-3, 1e-2, 1e-1],
    "iteration_limits": [10, 50],
}

def generate_array_from_path(path: str) -> Dict:
    video = cv2.VideoCapture(path)
    if not video.isOpened():
        raise FileNotFoundError(f"La vidéo n'a pas été ouverte")

    framesNumber = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    video_buffer = np.empty((framesNumber, frameHeight, frameWidth), np.float32)

    count = 0
    while True:
        keep_going, new_frame = video.read()
        
        if not keep_going or new_frame is None:
            break
        if count < framesNumber:
            video_buffer[count] = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)/255.0
            count += 1
        else:
            break
    return {"Video_content": video_buffer, "fps": fps, "height": frameHeight, "width": frameWidth}

bus_fight, falling_ball, vapeur = (generate_array_from_path(path) for path in ["./bus_fight.mp4", "./falling_ball.mp4", "./vapeur.mp4"])

def lucas_kanade_sparse(video_path, max_corners=100):
    """
    Flot optique sparse avec Lucas-Kanade.
    Suit des points d"interet entre frames successives.
    """
    cap = cv2.VideoCapture(video_path)
    # Parametres pour la detection de coins
    feature_params = dict(
    maxCorners=max_corners,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
    )
    # Parametres Lucas-Kanade
    lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) #type: ignore
    trajectories = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calcul du flot optique
        p1, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) #type: ignore
        # Selection des bons points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]
            # Stockage des deplacements
            for new, old in zip(good_new, good_old):
                trajectories.append({
                    "old": old.flatten(),
                    "new": new.flatten(),
                    "flow": (new - old).flatten()
                })
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    
    cap.release()
    return trajectories

def lucas_kanade_benchmark():
    lucas_kanade_sparse("./falling_ball.mp4")

def generate_videos_by_gradient(video: Dict, parameters: Dict[str, List[float|int]], normL2: bool):
    for alpha_squared in parameters["alphas"]:
        for step in parameters["steps"]:
            for MaxIter in parameters["iteration_limits"]:
                counts = np.zeros(MaxIter) #type: ignore
                if video is bus_fight:
                    video_name = "bus_fight"
                elif video is falling_ball:
                    video_name = "falling_ball"
                elif video is vapeur:
                    video_name = "vapeur"
                r.print(f"Résolution de {video_name} par gradient avec paramètres alpha squared : {alpha_squared}, pas: {step}, itérations max : {MaxIter}")
                output_name = f"tests/Norm_L2/gradient_results/{video_name}_{alpha_squared}_{step}_{MaxIter}.mp4" if normL2 else f"tests/Norm_L1/gradient_results/{video_name}_{alpha_squared}_{step}_{MaxIter}.mp4"
                
                #Création du dossier si le chemin n'existe pas déjà
                if not os.path.exists(os.path.split(output_name)[0]):
                    os.makedirs(os.path.split(output_name)[0])
                
                optical_flow_x, optical_flow_y, counts  = horn_schunck_rs.solve_gradient_descent(video["Video_content"], alpha_squared, step, MaxIter, 1e-3, normL2) #type: ignore

                output = cv2.VideoWriter(f"{output_name}", cv2.VideoWriter_fourcc(*"mp4v"), video["fps"], (video["width"], video["height"]), isColor=False) #type: ignore
                for frame_x, frame_y in zip(optical_flow_x, optical_flow_y):
                    magnitude = frame_x**2 + frame_y**2
                    detection = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #type: ignore
                    
                    output.write(detection)
                output.release()
                r.print(f"Fichier {output_name} écrit.", end="\n\n")

def gradient_benchmark():
    horn_schunck_rs.solve_gradient_descent(falling_ball["Video_content"], 1, 1e-3, 50, 1e-8, True)

def generate_videos_by_gauss_seidel(video: Dict, parameters: Dict[str, List[float|int]]):
    for alpha_squared in parameters["alphas"]:
        for MaxIter in parameters["iteration_limits"]:
            if video is bus_fight:
                video_name = "bus_fight"
            elif video is falling_ball:
                video_name = "falling_ball"
            elif video is vapeur:
                video_name = "vapeur"
            output_name = f"tests/Norm_L2/gauss_seidel_results/{video_name}_{alpha_squared}_{MaxIter}.mp4"
            
            if not os.path.exists(os.path.split(output_name)[0]):
                os.makedirs(os.path.split(output_name)[0])
            r.print(f"Résolution de {video_name} par Gauss-Seidel avec paramètres alpha squared : {alpha_squared}, itérations max : {MaxIter}")
            optical_flow_x, optical_flow_y = horn_schunck_rs.solve_gauss_seidel(video["Video_content"], alpha_squared, MaxIter) #type: ignore
            output = cv2.VideoWriter(f"{output_name}", cv2.VideoWriter_fourcc(*"mp4v"), video["fps"], (video["width"], video["height"]), isColor=False) #type: ignore

            for frame_x, frame_y in zip(optical_flow_x, optical_flow_y):
                magnitude = frame_x**2 + frame_y**2
                detection = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #type: ignore
                
                output.write(detection)
            output.release()
            r.print(f"Fichier {output_name} écrit.", end="\n\n")

def gauss_seidel_benchmark():
    horn_schunck_rs.solve_gauss_seidel(falling_ball["Video_content"], 1, 50)

def benchmarks(repeat: int) -> Tuple[ndarray, ndarray, ndarray]:
        lucas_kanade_measured_times = timeit.repeat(lucas_kanade_benchmark, repeat=repeat, number=1)
        gradient_measured_times = timeit.repeat(gradient_benchmark, repeat=repeat, number=1)
        gauss_seidel_measured_times = timeit.repeat(gauss_seidel_benchmark, repeat=repeat, number=1)

        return np.array(lucas_kanade_measured_times), np.array(gradient_measured_times), np.array(gauss_seidel_measured_times)

def generate_videos():
    for video in bus_fight, falling_ball, vapeur:
        generate_videos_by_gauss_seidel(video, GS_parameters)
        generate_videos_by_gradient(video, gradient_parameters, True)
        generate_videos_by_gradient(video, gradient_parameters, False)


generate_videos()


# repeat = 5
# times = benchmarks(repeat)
# r.print(f"Moyenne de lucas-kanade : {times[0].mean()}\nMoyenne du gradient : {times[1].mean()}\nMoyenne de Gauss-Seidel : {times[2].mean()}\nSur {repeat} itérations.")
# r.print(f"Lucas-kanade a été en moyenne plus rapide de {times[1].mean()/times[0].mean()} comparé à la descente de gradient et {times[1].mean()/times[0].mean()} comparé à Gauss-Seidel.")