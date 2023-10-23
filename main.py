import sys
from Mediapipe_Body_vedio import MediapipeBodyVideo

if __name__ == "__main__":
    MediapipeBody_Video = MediapipeBodyVideo()
    MediapipeBody_Video.run_pose_estimation(sys.argv[1])
       