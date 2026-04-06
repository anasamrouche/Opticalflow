import os
from typing import List
import horn_schunck_rs
import numpy as np
import rich as r
import cv2

video_paths = {
    "pingpongsd": "./tests/pingpongsd.mp4",
    "pingpong": "./tests/pingpong.mp4",
    "pingponghd": "./tests/pingponghd.mp4"
}
alphas: List[float] = [0.1, 0.5, 1, 5, 10, 50]
steps: List[float] = [1e-5, 8e-5, 2e-4, 1e-3]
iters: List[int] = [15, 30, 50]

frameCount = 80

for video_name, path in video_paths.items():
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
    temp_buffer = np.empty((frameCount, frameHeight, frameWidth, 3), np.float32)
    video_buffer = np.empty((frameCount, frameHeight, frameWidth), np.float32)

    count = 0
    keep_going = True
    while count < frameCount and keep_going:
        keep_going, temp_buffer[count] = video.read()
        count += 1

    """
    Conversion de l'array représentant les couleurs en un array tenant seulement compte de la luminosité
    Il doit sûrement y avoir des formules plus élaborées et performantes pour calculer la luminosité d'un pixel en fonction
    des couleurs (me renseigner aussi dessus).
    """
    video_buffer = np.linalg.norm(temp_buffer, axis=3)
    r.print(f"Vidéo de taille {video_buffer.shape[1:]} et de longueur {video_buffer.shape[0]}")
    """
    Résolution par la méthode du gradient
    Et écriture de la vidéo obtenue
    """
    for alpha_squared in alphas:
        for step in steps:
            for MaxIter in iters:
                match video_name:
                    case "pingpongsd": output_name = f"tests/Norm_L1/gradient_results/low_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                    case "pingpong": output_name = f"tests/Norm_L1/gradient_results/standard_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                    case _: output_name = f"tests/Norm_L1/gradient_results/high_quality/{alpha_squared}_{step}_{MaxIter}.mp4"
                if not os.path.exists(os.path.split(output_name)[0]):
                    os.makedirs(os.path.split(output_name)[0])
                optical_flow_x, optical_flow_y = horn_schunck_rs.solve_gradient_descent(video_buffer, alpha_squared, step, MaxIter, False)
                r.print(f"Résolution par gradient avec paramètres alpha squared : {alpha_squared}, pas: {step}, itérations max : {MaxIter}")
                output = cv2.VideoWriter(f"{output_name}", cv2.VideoWriter_fourcc(*"mp4v"), fps, (frameWidth, frameHeight), isColor=False) #type: ignore

                for frame_x, frame_y in zip(optical_flow_x, optical_flow_y):
                    movement_detection = (255 * ((frame_x ** 2 + frame_y**2) > 0.5)).astype(np.uint8)
                    output.write(movement_detection)
                output.release()
                r.print(f"Fichier {output_name} écrit.")