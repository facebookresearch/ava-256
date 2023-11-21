
#codec_avatar_20210511_0825_row429_fov_full_order_by_frame_flasharray

#20210511-0825-row429

# in 28idx
#codec_avatar_20220315_0818_hdz165_fov_full_order_by_frame_flasharray
#codec_avatar_20211108_1309_ajd691_fov_full_order_by_frame_flasharray

ids = open('32ids.txt', 'rt').readlines()

for e in ids:
    toks = e.split('_')

    mcd = toks[2]
    mct = toks[3]
    msid = toks[4]

    fn = f"{mcd}-{mct}-{msid}"
    wfd=open(fn, 'wt')

    e = e.replace('flasharray', 'dec')

    wfd.write(e)
    wfd.close()
