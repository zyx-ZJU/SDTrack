from pathlib import Path
import torch 
import argparse 
import numpy as np
import cv2
import os

from tracker.tracker_zoo import create_tracker
from utils.common import non_max_suppression, ltwh2xyxy, xyxy2ltwh

from opts import opt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # repo root absolute path
EXAMPLES = FILE.parents[0]  # examples absolute path
WEIGHTS = EXAMPLES / 'weights'

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info

def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        bbox = ltwh2xyxy(bbox)
        if bbox[3] < min_height:
            continue
        detection = np.concatenate((bbox, np.array([confidence]), np.array([0]), feature)) #缺少cls信息，补充一下
        detection_list.append(detection)
    return detection_list

def new_tracker():
    tracking_config =\
    ROOT /\
    'CCtracking' /\
    'tracker' /\
    opt.tracking_method /\
    'configs' /\
    (opt.tracking_method + '.yaml')
    tracker = create_tracker(
        opt.tracking_method,
        tracking_config,
        opt.reid_model,
        opt.device,
        False
    )
    return tracker

def write_MOT17_results(output_file, results, frame_idx):
    
    bboxes = results[:, :4]
    bboxes = xyxy2ltwh(bboxes)

    outputs = np.concatenate((bboxes, results[:, 4:]), axis= 1)

    f = open(output_file, 'a')
    for output in outputs:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            frame_idx, output[4], output[0], output[1], output[2], output[3]),file=f)

def run(args):
    seq_info = gather_sequence_info(args.sequence_dir, args.detection_file)

    # create tracker
    tracker = new_tracker()
    
    min_frame_idx, max_frame_idx = seq_info["min_frame_idx"], seq_info["max_frame_idx"]

    for i in range(max_frame_idx):
        detections = create_detections(seq_info["detections"], i+1, args.min_detection_height) 
        if len(detections) == 0:
            continue

        # preprocessing
        detections = [d for d in detections if d[4] >= args.min_confidence]

        boxes = np.array([d[:4] for d in detections]) # in xyxy format
        scores = np.array([d[4] for d in detections])
        indices = non_max_suppression(
            boxes, args.nms_max_overlap, scores
        )
        
        detections = [detections[i] for i in indices]
        detections = np.array(detections)
        detections = torch.tensor(detections)
        
        # read raw image for ReID
        img = cv2.imread(seq_info['image_filenames'][i+1], cv2.IMREAD_COLOR)
        
        # tracking
        tracker_outputs = tracker.update(detections.cpu().detach(), img)
        
        # store results
        txt_path = os.path.join(args.output_file, seq_info['sequence_name']+'.txt')
        if len(tracker_outputs):
            write_MOT17_results(txt_path, tracker_outputs, i+1)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking_method', type=str, default='strongsort', help='deepocsort, botsort, strongsort, ocsort, bytetrack, sparsetrack')
    parser.add_argument('--reid_model', default=WEIGHTS / 'mobilenetv2_x1_4.pt', help="Path to the ReID model")
    parser.add_argument('--device', default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--sequence_dir', default='C:/Users/zhao/Desktop/project/MOT_challenge/MOT17/train/MOT17-02-FRCNN', help="Path to MOTChallenge sequence directory")
    parser.add_argument('--detection_file', default='data/MOT17_val_YOLOX+BoT/MOT17-02-FRCNN.npy', help="Path to custom detections")
    parser.add_argument('--output_file', default=EXAMPLES / 'runs', help="Path to the tracking output file. This file will contain the tracking results on completion")
    parser.add_argument('--min_confidence', default=0.6, type=float, help="Detection confidence threshold. Disregard all detections that have a confidence lower than this value.")
    parser.add_argument('--min_detection_height', default=0, type=int, help="threshold on the detection bounding box height. Detections with height smaller than this value are disregarded")
    parser.add_argument('--nms_max_overlap', default=1.0, type=float, help="Non-maxima suppression threshold: Maximum detection overlap.")
    parser.add_argument('--display', default=True, help="Show intermediate tracking results")
    opt = parser.parse_args()
    
    return opt

def main(opt):
    run(opt)

if __name__ == "__main__":
    opt = parse_opt()

    main(opt)