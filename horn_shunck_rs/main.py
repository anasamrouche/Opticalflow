import argparse
from typing import Dict, Tuple
import horn_schunck_rs
import cv2
import numpy as np
from numpy import ndarray
import os

def main():
    args = parse_arguments()
    if args.step_size == None and args.method == "gradient":
        raise ValueError("A step must be specified for gradient method")
    if args.method == "GS":
        try:
            alpha_squared = float(args.alpha_squared)
            IterMax = int(args.IterMax)
        except:
            raise ValueError("Conversion impossible")
    elif args.method == "gradient":
        try:
            alpha_squared = float(args.alpha_squared)
            IterMax = int(args.IterMax)
            step_size = float(args.step_size)
            tol = float(args.tol) if args.tol != None else 1e9
        except:
            raise ValueError("Conversion impossible")

    if args.output_path == None:
        raise ValueError("Chemin de sortie manquant")
    output_path = args.output_path
    try:
        video_array, metadata = convert_video(args.video_path)
        print(video_array.shape)
    except FileNotFoundError:
        print("Chemin de vidéo spécifié inconnu")
        raise FileNotFoundError

    match args.method:
        case "GS": optical_flow_u, optical_flow_v = horn_schunck_rs.solve_gauss_seidel(video_array, alpha_squared, IterMax)
        case "gradient": optical_flow_u, optical_flow_v, counts = horn_schunck_rs.solve_gradient_descent(video_array, alpha_squared, step_size, IterMax, tol, args.norm == "L2")
    
    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])

    output = cv2.VideoWriter(f"{output_path}", cv2.VideoWriter_fourcc(*"mp4v"), metadata["fps"], (metadata["frameWidth"], metadata["frameHeight"]), isColor=False) #type: ignore
    for frame_x, frame_y in zip(optical_flow_u, optical_flow_v):
        movement_detection = (255 * ((frame_x ** 2 + frame_y**2) > 0.1)).astype(np.uint8)
        output.write(movement_detection)
    output.release()
    print(f"Fichier {output_path} écrit.", end="\n\n")    
    


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog = "Horn-Schunk solver",
        description= "A solver detecting movements on videos based on iterative methods",
    )

    parser.add_argument(
        "--video_path", "-f",
        action="store",
        dest="video_path"
    )

    parser.add_argument(
        "--alpha_squared",
        action="store",
        dest="alpha_squared"
    )

    parser.add_argument(
        "--method", "-m",
        action="store",
        choices=["GS", "gradient"]
    )

    parser.add_argument(
        "--step_size",
        action="store",
        default=None
    )

    parser.add_argument(
        "--IterMax",
        action="store",
        dest="IterMax"
    )

    parser.add_argument(
        "--tol", "-t",
        action="store",
        dest="tol",
        default=None
    )

    parser.add_argument(
        "--norm", "-n",
        action="store",
        # choices=["L1, L2"],
        dest="norm"
    )

    parser.add_argument(
        "--output_path", "-o",
        action="store",
        dest="output_path"
    )

    return parser.parse_args()

def convert_video(video_path: str) -> Tuple[ndarray, Dict[str, int]]:
    frameCount = 80
    """
    Ouverture du fichier vidéo et enregistrement des paramètres
    """
    video = cv2.VideoCapture(video_path)
    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    """
    Représentation de la vidéo sous la forme d'un array numpy
    Il y a peut-être une méthode appelable pour s'en charger plutôt que de passer par une boucle for
    Il faudrait que je me renseigne
    """
    video_buffer = np.empty((frameCount, frameHeight, frameWidth), np.float32)

    count = 0
    keep_going = True
    while count < frameCount and keep_going:
        keep_going, new_frame = video.read()
        video_buffer[count] = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        count += 1
    
    return video_buffer, {
        "frameCount": int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        "frameWidth": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "frameHeight": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps
        }

if __name__ == "__main__":
    main()