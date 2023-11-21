
from care.data.io import typed
from care.strict.data.io.file_system.airstore_client import register_airstore_in_fsspec


if __name__ == "__main__":
    register_airstore_in_fsspec()

    capture = "1000275597227205"  # Second

    images_table_name = "codec_avatar_mgr_12k_frames_no_user_data"
    frame_id = "0"
    segment = "dynamic_range-of-motion-1"

    url = f"airstoreds://{images_table_name}/image?subject_id={capture}&frame_id={int(frame_id)}&segment={segment}"
    print(f"{url=}")

    # Hangs forever :(
    print("Hope I don't hang!")
    img = typed.load(url, extension="png")

    print(img)

    # I20231030 20:51:01.696367 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [12] after sleeping for 3.126307 seconds
    # I20231030 20:51:04.823659 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [13] after sleeping for 7.563880 seconds
    # I20231030 20:51:12.388401 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [14] after sleeping for 13.929574 seconds
    # I20231030 20:51:26.318758 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [15] after sleeping for 3.104173 seconds
    # I20231030 20:51:29.423913 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [16] after sleeping for 29.882892 seconds
    # I20231030 20:51:59.307569 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [17] after sleeping for 39.482822 seconds
    # I20231030 20:52:38.791707 2771868 DecryptionClient.cpp:159] Decrypt server not ready, retry [18] after sleeping for 99.942413 seconds
