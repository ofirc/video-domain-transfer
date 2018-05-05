# video-domain-transfer
Misc tools and automation scripts for face_swap, StarGAN and video editing.

# Overview
For each of the methods described in the "Unsupervised video to video translation" paper we
have the following scripts to run and evaluate the results:
| Approach                                                           | Script                    |
|--------------------------------------------------------------------|---------------------------|
| MSE evaluation of vanilla StarGAN                                  | face_swap_mse_eval.py     |
| Align input dataset to CelebA                                      | align_to_CelebA.py        |
| Aspect ratio resize + face swap                                    | resize_and_run_stargan.py |

We have tested our scripts only on a Windows machine but it should work on Linux as well.

## Dependencies
| Dependency                                        |
|---------------------------------------------------|
| [face swap](https://github.com/ofirc/face_swap/)  |
| [StarGAN](https://github.com/ofirc/StarGAN)       |
| [Anaconda](https://www.anaconda.com/)             |

## Usage
1. Open an Anaconda Prompt
2. MSE evaluation of vanilla StarGAN 

   `python face_swap_mse_eval.py --video video.mp4`

   `python face_swap_mse_eval.py --frames <frames dir>`
3. Align input dataset to CelebA

* Find average CelebA face

    Produces a CSV with average face landmarks (68 absolute x,y pairs)

   `python face_landmarks.py [max_num_frames]`
* Align input frames to average face landmarks

   `python align_to_CelebA.py
  --average_face_file mean_CelebA_1000_samples_952_actual_2018-02-21_19_2_1.csv
  --frames c:\stargan_results\orig_frames`
4. Aspect ratio resize + face swap

   `python resize_and_run_stargan.py --input_video <input video> --output_video [<output video>]`
  
