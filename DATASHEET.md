Datasheet in the format of "Datasheets for datasets" as described in

> Gebru, Timnit, et al. "Datasheets for datasets." Communications of the ACM 64.12 (2021): 86-92.

# Ava-256 Dataset

<!-- TODO(julieta) add brief summary here, bibtex -->


## Motivation

1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*

    This dataset was created to to facilitate the study of both the construction of high quality avatars, as well as their driving from limited sensing -- in this case, from the infrarred cameras in a [Quest Pro](https://www.meta.com/ca/quest/quest-pro/), a commercially available headset.


1. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**

    The dataset was created by the Codec Avatars Team within Meta Reality Labs, at Meta.


1. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*

    Meta Platforms Inc.


1. **Any other comments?**

    None.





## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*

    Each instance comprises two captures of the same person: one in a high resolution dome, and one captured from a Quest Pro headset.
    We also provide assets used to faciliate the creation of photorealistic avatars.


1. **How many instances are there in total (of each type, if appropriate)?**

    Our dataset constains 256 paired subject captures, each containing a dome capture and a headset capture.


1. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*

    The dataset contains samples of human appearance. We collected self-reported demographic data about age, gender, and skin tone. First, we balanced the data by gender, which resulted in
    * 124 subjects who identified as men or male,
    * 124 subjects who identified as women or female,
    * 6 subjects who identified as non-binary, third gender, or another gender,
    * 2 subjects who chose not to respond this question.

    Overall, our dataset is biased towards lighter skinned people under 35 years old.
    Collecting a wider diversity of subjects is something we are actively pursuing.


1. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*

    The detailed structure of each of our paired captures is, for each pair:
    ```
    m--{capture.mcd}--{capture.mct}--{capture.sid}--GHS
    .
    ├── decoder
    │   ├── camera_calibration.json
    │   ├── frame_list.csv
    │   ├── head_pose
    │   │   └── head_pose.zip
    │   ├── image
    │   │   ├── cam{cam_name_01}.zip
    │   │   ├── cam{cam_name_02}.zip
    │   │   ├── ...
    │   │   └── cam{cam_name_80}.zip
    │   ├── background_image [4TB only]
    │   │   └── background_image.zip
    │   ├── expression_codes [4TB only]
    │   │   └── aeparams_1440000.pkl
    │   ├── keypoints_3d
    │   │   └── keypoints_3d.zip
    │   ├── kinematic_tracking
    │   │   ├── registration_vertices_mean.npy
    │   │   ├── registration_vertices_variance.txt
    │   │   └── registration_vertices.zip
    │   ├── segmentation_parts
    │   │   ├── cam{cam_name_01}.zip
    │   │   ├── cam{cam_name_02}.zip
    │   │   ├── ...
    │   │   └── cam{cam_name_80}.zip
    │   └── uv_image
    │      ├── color_mean.png
    │      ├── color_variance.txt
    │        └── color.zip
    └── encoder
        ├── frame_list.csv
        └── image
            ├── cam-cyclop.zip
            ├── cam-left-eye-atl-temporal.zip
            ├── cam-left-mouth.zip
            ├── cam-right-eye-atl-temporal.zip
            └── cam-right-mouth.zip
    ```

    The `decoder` folder contains data and assets for high quality avatar reconstruction.

    * `decoder/camera_calibration.json`: is a json file with the intrinsic and extrinsic paramters of each camera.
    * `decoder/frame_list.csv` is a csv with `N_FRAMES` rows (1 per frame) and two columns: `seg_id` and `frame_id`;
       the former is the name of the segment, and the latter is the frame number. `N_FRAMES` is the number of frames in the dataset, and is either ~5,000 or ~10,000 depending on our release.
    * `decoder/head_pose/head_pose.zip` contains a `N_FRAMES` 3x4 transforms with the pose of the head of the person being captured per frame, in world coordinates.
    * `decoder/image/cam{cam_name_XX}.zip` refers to a folder with `N_CAMERAS` uncompressed zip files. Each uncompressed zip file has `N_FRAMES` images corresponding to the image taken by a certain camera at a certain frame.
    In our releases, `N_CAMERAS` is 80, and `N_FRAMES` is either around 5,000 or 10,000.
    The images follow the naming convention `{frame_id:06d}.avif`, as indicated in `frame_list.csv`, and are stored in avif format.
    The images are RGB, with resolution 1024 x 667 or 2048 x 1334 depending on the release package.
    * `decoder/background_image/background_image.zip` is a zip file with `N_CAMERAS` images that contain the empty background (without person) for each viewpoint which is helpful for background removal. 
   Background images are stored as `{cam_name}.avif`. Only available for the 4TB version of the dataset.
    * `decoder/expression_codes/aeparams_1440000.pkl` is a pickled dictionary containing a mapping `frame_id => expression_code` where `expression_code` is a 256-dimensional code describing the person's facial expression at `frame_id`. Only available for the 4TB version of the dataset.
    * `decoder/keypoints_3d/keypoints_3d.zip` is a zip file with `N_FRAMES` files. Each file follows the format `{frame_id:06d}.npy`, and is a npy file with a matrix with up to 274 rows 6 rows representing 3d keypoints for each frame.
    Note that some keypoints may not be present if the number of inliers falls below a certain thresold. The format of each row is:
        ```
        | Column | Entry                                |
        | -----: | :----------------------------------- |
        |      0 | Keypoint ∈ {0, 1, ..., 273}          |
        |      1 | x                                    |
        |      2 | y                                    |
        |      3 | z                                    |
        |      4 | Sum of confidence values for inliers |
        |      5 | Number of inliers                    |
        ```
    * `decoder/segmentation_parts/cam{cam_name_XX}.zip` is a folder with `N_CAMERAS` zip files.
    Each zip file has `N_FRAMES` images with the format `{frame_id:06d}.png`, according to the frames indicated in `frame_list.csv`.
    The images are segmentation images with the following values:
        ```
        | Number | Class         |
        | -----: | :------------ |
        | 1      | Hair          |
        | 2      | Body          |
        | 3      | Chair         |
        | 4      | Apparel       |
        | 5      | Lenses        |
        | 6      | UpperLip      |
        | 7      | LowerLip      |
        | 8      | UpperTeeth    |
        | 9      | LowerTeeth    |
        | 10     | TongueVisible |
        ```
    * `decoder/uv_image/color_mean.png` is a 1000 x 1000 RGB images with the mean of the uv texture maps of this capture.
    * `decoder/uv_image/color_variance.txt` contains a scalar that is the variance of the uv texture maps of this capture.
    * `decoder/uv_image/color.zip` is a zip file with `N_FRAMES` RGB 1000 x 1000 images, each containing the uv texture map of the subject in each frame.
    The name of these files is `{frame_id:06}.avif`, according to the frames in `frame_list.csv`, and the images are in avif format.


    Meanwhile the `encoder` folder contains data and assets for high quality avatar reconstruction.

    * `encoder/frame_list.csv` is a csv with `N_FRAMES` rows (1 per frame) and two columns: `seg_id` and `frame_id`;
      the former is the name of the segment, and the latter is the frame number. `N_FRAMES` is the number of frames in the encoder capture.
    * `encoder/image/cam-XX.zip`: are 5 zip files corresponding to the 5 infrarred cameras of a Quest Pro.
      Each zip file has `N_FRAMES` images with format `{frame_id:06d}.avif`, following `frame_list.csv`.



1. **Is there a label or target associated with each instance? If so, please provide a description.**

    No.


1. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*

    Not all images listed in the `frame_list.csv` are available; about 0.1% are missing due to system failures during capture. These so-called "dropped" frames are a common occurrence in large-scale capture systems.


1. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*

    Each subject is capture twice, thus the captures are _paired_ in the sense that they belong to the likeness of the same subject.
    The scripts used for both captures are also similar, although not identical.
    This relationship is explicit, since the data of each subject is provided under the same folder.


1. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*

    To study single-subject reconstruction, we recommend researchers withhold the `EXP_free_face` segment for validation, which has varied and extreme motion by the subject.

    To study multi-subject reconstruction and generalization to new identities, we recommend users withhold 20 subjects for validation, and reconstruct them from sparse inputs.

    To study avatar driving, we recommend users withhold 12 subjects, as we do in the examples in our repo.


1. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*

    Besides a small percentage of missing images, assets are naturally noisy, since they are computed automatically. For example, segmentations might be imperfect, and kinematic tracking is not 100% accurate.


1. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*

    The dataset is self-contained.


1. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*

    No.


1. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*

    No.


1. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*

    Yes, the dataset consists of captures of people.


1. **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*

    We only provide aggregated statistics of gender, skin tone, and age. Please refer to question 3 for a more detailed breakdown.


1. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*

    Yes, by looking at the provided images of the subjects.


1. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*

    No.


1. **Any other comments?**

    None.





## Collection Process


1. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*

    The subjects were captured in a high resolution dome and wearing a headset.


1. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*

    We used a high resolution dome with 172 cameras.
    Each camera takes images at 30 fps, with a resolution of 4096 x 2668.
    The subjects were directed by a research assistant to express a range of emotions, make face deformations, and say phrases that comprise phonemes commonly occurring in English.
    The headset captures were done in a separate room, with a Quest Pro headset augmented with 5 additional infrarred cameras.


1. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**

    See answer to question #3 in [Composition](#Composition).


1. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**

    Subjects were at least 18 years old at the time of capture, provided their informed consent in writing, were compensated at a rate of USD $50 per hour, rounded to the next half hour.


1. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*

    The data was collected over 25 months, between August of 2021 and September of 2023 in Pittsburgh, PA.


1. **Were any ethical review processes conducted (e.g., by an institutional review board)?** *(If so, please provide a description of these review processes, including the outcomes, as well as a link or other access point to any supporting documentation.)*

    We followed an internal research review that includes ethical considerations.
    These reviews are internal.


1. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*

    Yes; the dataset consists of captures of people.


1. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**

    From the individuals directly.


1. **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*

    Yes, they provided their informed written consent and were compensated for their time.


1. **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*

    Yes, the individuals gave their informed consent in writing for both capture and distribution of the data.


1. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*

    Before and during the capture, subjects were informed that they could drop out of the study at any time, and still be compensated for their time.


1. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*

    No.
+

1. **Any other comments?**

    None.





## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*

    Please refer to question #4 in [Composition](#composition) for details on the derived assets provided as part of the dataset.


1. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*

    Yes. The original raw capture data takes up several petabytes, and is not distributed due to the infeasibility of distributing such large amounts of data.


1. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*

    No; the methods used for kinematic tracking, keypoint extraction and segmentation are internal.


1. **Any other comments?**

    None.





## Uses


1. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*

    Yes, we provide code and checkpoints to both build a multi-person avatar model with a consistent expression space, as well as a generalizable driver to control the avatar from headset images.


1. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*

    Not yet, as publications use this dataset we will update the github repository.


1. **What (other) tasks could the dataset be used for?**

    The dome captures could be used to study the generation of novel avatars from sparse inputs (eg, single, or few views), towards the fast creation of high quality avatars for people not in the training set.


1. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*

    None to our knowledge.


1. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*

    No.


1. **Any other comments?**

    None.




## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*

    Yes, the dataset is publicly available.


1. **How will the dataset be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*

    The dataset can be downloaded from AWS. We provide an easy python script to download the data: https://github.com/facebookresearch/ava-256/blob/main/download.py.


1. **When will the dataset be distributed?**

    The dataset is publicly available as of Wednesday June 12, 2024.


1. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*

    The dataset is licensed under a the CC-by-NC 4.0 license.


1. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*

    Not to our knowledge.


1. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*

    Not to our knowledge.


1. **Any other comments?**

    None.





## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**

    The open source team of the Codec Avatars Lab is supporting the dataset.
    The dataset is hosted on an AWS S3 bucket paid for by Meta.


1. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**

    You may contact Julieta Martinez at julietamartinez@meta.com.


1. **Is there an erratum?** *(If so, please provide a link or other access point.)*

    Currently, no. If we encounter errors, future versions of the dataset may be released, and will be versioned.
    The new versions will be provided in the same location.


1. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*

    Same as previous.


1. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*

    No.


1. **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*

    Yes; all data will be versionedand maintained by the same team.


1. **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*

    It is unlikely other labs will be able to produce a dataset in a similar format in the near future:
    the general lack of accessibility for this kind of data is part of our motivation for releasing our dataset.
    However, if that happens we would welcome the collaboration to create larger publicly available datasets in the future.


1. **Any other comments?**

    None.
